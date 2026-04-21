"""Abstract base class for all agents in the tutoring system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import time


@dataclass
class AgentMessage:
    """Structured message for inter-agent communication."""
    sender: str
    receiver: str
    msg_type: str          # "request" | "response" | "broadcast"
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class BaseAgent(ABC):
    """
    Base class for all agents. Defines the communication protocol and
    lifecycle interface used by the orchestrator.
    """

    def __init__(self, agent_id: str, tools: Optional[List[Any]] = None):
        self.agent_id = agent_id
        self.tools = {t.name: t for t in (tools or [])}
        self._inbox: List[AgentMessage] = []
        self._outbox: List[AgentMessage] = []
        self._step_count = 0
        self._is_active = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self) -> None:
        """Reset agent state for a new episode."""

    @abstractmethod
    def act(self, observation: np.ndarray, context: Dict[str, Any]) -> Any:
        """
        Select an action given the current observation and shared context.
        Returns an agent-specific action object.
        """

    def step_done(self, feedback: Dict[str, Any]) -> None:
        """Called by orchestrator after environment step with transition feedback."""

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def send(self, receiver: str, msg_type: str, payload: Dict) -> None:
        self._outbox.append(
            AgentMessage(sender=self.agent_id, receiver=receiver, msg_type=msg_type, payload=payload)
        )

    def receive(self, message: AgentMessage) -> None:
        self._inbox.append(message)

    def flush_outbox(self) -> List[AgentMessage]:
        msgs, self._outbox = self._outbox, []
        return msgs

    def process_inbox(self) -> List[AgentMessage]:
        msgs, self._inbox = self._inbox, []
        return msgs

    # ------------------------------------------------------------------
    # Tool use
    # ------------------------------------------------------------------

    def use_tool(self, tool_name: str, **kwargs) -> Any:
        if tool_name not in self.tools:
            raise KeyError(f"Agent {self.agent_id} has no tool '{tool_name}'")
        return self.tools[tool_name](**kwargs)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id, "steps": self._step_count}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id})"
