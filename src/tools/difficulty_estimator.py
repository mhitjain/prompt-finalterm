"""
Difficulty Estimator Tool — custom tool implementing Item Response Theory (IRT)
to provide principled difficulty estimates for questions.

The 2-Parameter Logistic (2PL) IRT model:
    P(X=1 | θ, b, a) = 1 / (1 + exp(-a(θ - b)))

where:
  θ (ability): latent student ability estimated from performance history
  b (difficulty): item difficulty parameter
  a (discrimination): how strongly the item differentiates ability levels

Ability estimation uses Expected A Posteriori (EAP) integration over
a normal prior N(0, 1).

This tool enables the agents to set difficulty parameters that optimally
challenge the student (Vygotsky's Zone of Proximal Development).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import integrate, stats


class DifficultyEstimatorTool:
    """
    IRT-based difficulty estimator.

    Provides:
      - estimate_ability(): infer student θ from response vector
      - optimal_difficulty(): ZPD-optimal item difficulty for given ability
      - expected_score(): predicted probability of correct answer
      - calibrate_item(): estimate item parameters from response data
    """

    name = "difficulty_estimator"

    def __init__(self, n_quadrature: int = 21):
        self.n_q = n_quadrature  # quadrature points for EAP
        # Gauss-Hermite quadrature nodes and weights on N(0,1)
        self._theta_grid = np.linspace(-4, 4, n_quadrature)
        self._prior = stats.norm.pdf(self._theta_grid)
        self._prior /= self._prior.sum()

    def __call__(self, **kwargs):
        method = kwargs.pop("method", "optimal_difficulty")
        return getattr(self, method)(**kwargs)

    # ------------------------------------------------------------------
    # IRT model
    # ------------------------------------------------------------------

    @staticmethod
    def _2pl(theta: float, b: float, a: float = 1.5) -> float:
        """2PL probability of correct response."""
        return 1.0 / (1.0 + np.exp(-a * (theta - b)))

    # ------------------------------------------------------------------
    # Ability estimation (EAP)
    # ------------------------------------------------------------------

    def estimate_ability(
        self,
        responses: List[bool],
        difficulties: List[float],
        discrimination: float = 1.5,
    ) -> Tuple[float, float]:
        """
        Estimate student ability θ using Expected A Posteriori (EAP).

        Parameters
        ----------
        responses    : list of bool (True=correct, False=incorrect)
        difficulties : corresponding item difficulty parameters
        discrimination : IRT discrimination parameter a

        Returns
        -------
        (theta_hat, posterior_std) — mean and std of posterior
        """
        if not responses:
            return 0.0, 1.0

        # Compute likelihood at each quadrature point
        log_likelihood = np.zeros(self.n_q)
        for r, b in zip(responses, difficulties):
            p = self._2pl(self._theta_grid, b, discrimination)
            p = np.clip(p, 1e-9, 1 - 1e-9)
            log_likelihood += np.log(p) if r else np.log(1 - p)

        # Posterior = prior × likelihood
        log_posterior = np.log(self._prior + 1e-300) + log_likelihood
        log_posterior -= log_posterior.max()  # numerical stability
        posterior = np.exp(log_posterior)
        posterior /= posterior.sum()

        theta_hat = float(np.dot(self._theta_grid, posterior))
        theta_var = float(np.dot((self._theta_grid - theta_hat) ** 2, posterior))
        return theta_hat, float(np.sqrt(theta_var))

    # ------------------------------------------------------------------
    # Optimal difficulty (Zone of Proximal Development)
    # ------------------------------------------------------------------

    def optimal_difficulty(
        self,
        theta: float,
        target_success_rate: float = 0.70,
        discrimination: float = 1.5,
    ) -> float:
        """
        Compute the item difficulty b that yields the target success probability
        for a student with ability θ.

        P(correct) = target  ↔  b = θ - log(p/(1-p)) / a
        """
        p = np.clip(target_success_rate, 0.01, 0.99)
        b = theta - np.log(p / (1 - p)) / discrimination
        return float(np.clip(b, -3.0, 3.0))

    # ------------------------------------------------------------------
    # Fisher Information (for question selection)
    # ------------------------------------------------------------------

    def fisher_information(
        self, theta: float, b: float, a: float = 1.5
    ) -> float:
        """
        Fisher information of an item at ability θ.
        Higher information → item better discriminates at this ability level.
        I(θ) = a² * P(θ)(1 - P(θ))
        """
        p = self._2pl(theta, b, a)
        return float(a ** 2 * p * (1 - p))

    def most_informative_difficulty(
        self, theta: float, candidate_bs: List[float] = None
    ) -> float:
        """Return the difficulty b that maximises Fisher information at θ."""
        if candidate_bs is None:
            candidate_bs = list(np.linspace(-2, 2, 50))
        info = [self.fisher_information(theta, b) for b in candidate_bs]
        return float(candidate_bs[int(np.argmax(info))])

    # ------------------------------------------------------------------
    # Expected score prediction
    # ------------------------------------------------------------------

    def expected_score(
        self,
        knowledge: float,
        difficulty_normalised: float,
        discrimination: float = 1.5,
    ) -> float:
        """
        Predict probability of correct answer given [0,1] knowledge and difficulty.
        Maps to IRT scale internally.
        """
        theta = (knowledge - 0.5) * 4.0    # scale [0,1] → [-2, 2]
        b     = (difficulty_normalised - 0.5) * 4.0
        return float(self._2pl(theta, b, discrimination))

    def calibrate_item(
        self, responses: List[bool], abilities: List[float]
    ) -> Dict[str, float]:
        """
        MLE estimate of item parameters (b, a) from a set of student responses.
        Uses scipy minimise.
        """
        from scipy.optimize import minimize

        def neg_loglik(params):
            b, log_a = params
            a = np.exp(log_a)
            p = np.array([self._2pl(th, b, a) for th in abilities])
            p = np.clip(p, 1e-9, 1 - 1e-9)
            r = np.array(responses, dtype=float)
            return -float(np.sum(r * np.log(p) + (1 - r) * np.log(1 - p)))

        result = minimize(neg_loglik, x0=[0.0, 0.0], method="Nelder-Mead")
        b_est, log_a_est = result.x
        return {"b": float(b_est), "a": float(np.exp(log_a_est)), "converged": result.success}
