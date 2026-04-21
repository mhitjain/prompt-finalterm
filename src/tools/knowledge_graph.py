"""
Knowledge Graph Tool — custom tool that encodes topic prerequisite structure
and provides graph-theoretic utilities for curriculum planning.

The directed acyclic graph (DAG) captures:
  - Prerequisite dependencies between topics
  - Conceptual similarity weights (for transfer learning analysis)
  - Optimal learning ordering via topological sort
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple


TOPIC_NAMES = [
    "Basic Arithmetic",
    "Algebra Basics",
    "Geometry Basics",
    "Statistics Basics",
    "Linear Algebra",
    "Calculus Intro",
    "Probability",
    "Advanced Statistics",
    "ML Basics",
    "Deep Learning",
]

# (source, target, similarity_weight)
EDGES: List[Tuple[int, int, float]] = [
    (0, 1, 0.8),  # Arithmetic → Algebra
    (0, 2, 0.7),  # Arithmetic → Geometry
    (0, 3, 0.6),  # Arithmetic → Statistics
    (1, 4, 0.9),  # Algebra → Linear Algebra
    (1, 5, 0.85), # Algebra → Calculus
    (3, 6, 0.8),  # Statistics → Probability
    (6, 7, 0.9),  # Probability → Advanced Statistics
    (4, 8, 0.85), # Linear Algebra → ML Basics
    (5, 8, 0.80), # Calculus → ML Basics
    (6, 8, 0.75), # Probability → ML Basics
    (8, 9, 0.95), # ML Basics → Deep Learning
]


class KnowledgeGraphTool:
    """
    Custom tool providing knowledge graph services to the orchestrator.

    Capabilities:
      - best_next_topic(): recommend next topic to study
      - prerequisites_met(): check if a topic is unlocked
      - topological_order(): optimal study ordering
      - transfer_potential(): similarity between topics (for transfer learning)
    """

    name = "knowledge_graph"

    def __init__(self, n_topics: int = 10):
        self.n_topics = n_topics
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(n_topics))
        for src, tgt, w in EDGES:
            self.G.add_edge(src, tgt, weight=w)
        # Reverse graph for predecessor lookups
        self._rev = self.G.reverse()

    def __call__(self, **kwargs):
        method = kwargs.pop("method", "best_next_topic")
        return getattr(self, method)(**kwargs)

    # ------------------------------------------------------------------
    # Primary services
    # ------------------------------------------------------------------

    def best_next_topic(
        self, knowledge: List[float], mastery_threshold: float = 0.85
    ) -> int:
        """
        Recommend the highest-value next topic to study.

        Selection criteria (in priority order):
          1. All prerequisites must be met (knowledge ≥ 0.5)
          2. Maximise: (1 - current_knowledge) × prereq_readiness
             i.e., largest learning opportunity given readiness
        """
        knowledge = np.asarray(knowledge)
        mastered = set(np.where(knowledge >= mastery_threshold)[0])

        best_topic, best_score = 0, -1.0
        for topic in range(self.n_topics):
            if topic in mastered:
                continue
            prereqs = list(self._rev.successors(topic))
            if not prereqs:
                prereq_readiness = 1.0
            else:
                prereq_readiness = float(np.mean([knowledge[p] for p in prereqs]))
                if any(knowledge[p] < 0.5 for p in prereqs):
                    continue   # skip locked topic

            opportunity = (1.0 - knowledge[topic]) * prereq_readiness
            if opportunity > best_score:
                best_score = opportunity
                best_topic = topic

        return int(best_topic)

    def prerequisites_met(
        self, topic: int, knowledge: List[float], threshold: float = 0.5
    ) -> bool:
        prereqs = list(self._rev.successors(topic))
        return all(knowledge[p] >= threshold for p in prereqs)

    def topological_order(self) -> List[int]:
        return list(nx.topological_sort(self.G))

    def transfer_potential(self, source_topic: int, target_topic: int) -> float:
        """
        Compute knowledge transfer potential from source to target topic.
        Higher weight edges indicate more transferable knowledge.
        """
        try:
            paths = list(nx.all_simple_paths(self.G, source_topic, target_topic, cutoff=3))
            if not paths:
                return 0.0
            # Max path weight
            max_weight = max(
                np.prod([self.G[u][v]["weight"] for u, v in zip(p, p[1:])])
                for p in paths
            )
            return float(max_weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 0.0

    def locked_topics(self, knowledge: List[float], threshold: float = 0.5) -> List[int]:
        knowledge = np.asarray(knowledge)
        return [
            t for t in range(self.n_topics)
            if not self.prerequisites_met(t, knowledge.tolist(), threshold)
        ]

    def topic_centrality(self) -> Dict[int, float]:
        """PageRank centrality — more central topics are higher-value foundations."""
        return nx.pagerank(self.G)

    def visualise_graph(self) -> str:
        """Return adjacency list as string for logging."""
        lines = ["Knowledge Graph:"]
        for n in range(self.n_topics):
            successors = list(self.G.successors(n))
            if successors:
                snames = [TOPIC_NAMES[s] for s in successors]
                lines.append(f"  {TOPIC_NAMES[n]} → {', '.join(snames)}")
        return "\n".join(lines)
