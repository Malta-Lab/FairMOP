"""
Pareto frontier identification and multi-objective analysis.

Implements dominance-based filtering to extract the empirical Pareto
frontier from a set of evaluated configurations in the fairness-utility
space.

The Pareto frontier represents the set of non-dominated configurations,
i.e., configurations for which no other configuration simultaneously
improves both fairness and utility.

Algorithm (from the FairMOP methodology):

    For each configuration x_i:
        is_dominated ← false
        For each configuration x_j (j ≠ i):
            If (u_j ≥ u_i AND f_j ≥ f_i) AND (u_j > u_i OR f_j > f_i):
                is_dominated ← true
                break
        If NOT is_dominated:
            Add x_i to the Pareto set
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ConfigurationPoint:
    """A single point in the fairness-utility space.

    Attributes:
        config_name: Identifier for the hyperparameter configuration.
        utility: Utility metric value (higher is better).
        fairness: Fairness metric value (higher is better).
        hyperparams: The hyperparameters used for this configuration.
        metrics: All raw metric values for this configuration.
        is_pareto: Whether this point is on the Pareto frontier.
    """

    config_name: str
    utility: float
    fairness: float
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    is_pareto: bool = False


@dataclass
class ParetoResult:
    """Result of Pareto frontier analysis.

    Attributes:
        all_points: All evaluated configuration points.
        pareto_points: Only the Pareto-optimal points.
        utility_metric_name: Name of the utility metric used.
        fairness_metric_name: Name of the fairness metric used.
    """

    all_points: List[ConfigurationPoint]
    pareto_points: List[ConfigurationPoint]
    utility_metric_name: str = "utility"
    fairness_metric_name: str = "fairness"

    @property
    def n_configurations(self) -> int:
        return len(self.all_points)

    @property
    def n_pareto(self) -> int:
        return len(self.pareto_points)

    def summary(self) -> str:
        lines = [
            f"Pareto Analysis Summary",
            f"  Total configurations : {self.n_configurations}",
            f"  Pareto-optimal       : {self.n_pareto}",
            f"  Utility metric       : {self.utility_metric_name}",
            f"  Fairness metric      : {self.fairness_metric_name}",
        ]
        if self.pareto_points:
            lines.append("  Pareto configurations:")
            for p in self.pareto_points:
                lines.append(
                    f"    - {p.config_name}: "
                    f"utility={p.utility:.4f}, fairness={p.fairness:.4f}"
                )
        return "\n".join(lines)


def find_pareto_frontier(
    points: List[ConfigurationPoint],
) -> ParetoResult:
    """Identify the Pareto frontier from a set of configuration points.

    Implements the dominance-based filtering algorithm described in the
    FairMOP methodology (Algorithm 1).

    A configuration x_i dominates x_j if:
        - utility(x_i) ≥ utility(x_j) AND fairness(x_i) ≥ fairness(x_j)
        - with at least one strict inequality.

    Parameters:
        points: List of configuration points with utility and fairness values.

    Returns:
        A :class:`ParetoResult` with all points and the Pareto-optimal subset.
    """
    n = len(points)
    pareto_mask = [True] * n

    for i in range(n):
        if not pareto_mask[i]:
            continue
        for j in range(n):
            if i == j or not pareto_mask[j]:
                continue

            # Check if j dominates i
            u_i, f_i = points[i].utility, points[i].fairness
            u_j, f_j = points[j].utility, points[j].fairness

            if (u_j >= u_i and f_j >= f_i) and (u_j > u_i or f_j > f_i):
                pareto_mask[i] = False
                break

    pareto_points = []
    for i, is_pareto in enumerate(pareto_mask):
        points[i].is_pareto = is_pareto
        if is_pareto:
            pareto_points.append(points[i])

    # Sort Pareto points by utility for clean visualization
    pareto_points.sort(key=lambda p: p.utility)

    return ParetoResult(
        all_points=points,
        pareto_points=pareto_points,
    )


def build_pareto_from_results(
    results: Dict[str, Any],
    utility_metric: str = "avg_clip_score",
    fairness_metric: str = "gender_entropy",
    utility_invert: bool = False,
    fairness_invert: bool = False,
) -> ParetoResult:
    """Build Pareto analysis from FairMOP results dictionary.

    Parameters:
        results: Results dict with ``"configurations"`` key.
        utility_metric: Key in ``aggregates`` for utility metric.
        fairness_metric: Key in ``aggregates`` for fairness metric.
        utility_invert: If True, invert the utility value (for metrics like
                        FID where lower is better).
        fairness_invert: If True, invert the fairness value.

    Returns:
        A :class:`ParetoResult` instance.
    """
    points = []

    for config_name, config_data in results.get("configurations", {}).items():
        aggregates = config_data.get("aggregates", {})

        utility_val = aggregates.get(utility_metric, 0.0)
        fairness_val = aggregates.get(fairness_metric, 0.0)

        if utility_invert and utility_val > 0:
            utility_val = 1.0 / utility_val
        if fairness_invert and fairness_val > 0:
            fairness_val = 1.0 / fairness_val

        # Handle inf
        if utility_val == float("inf"):
            utility_val = 0.0
        if fairness_val == float("inf"):
            fairness_val = 0.0

        point = ConfigurationPoint(
            config_name=config_name,
            utility=utility_val,
            fairness=fairness_val,
            metrics=aggregates,
        )
        points.append(point)

    result = find_pareto_frontier(points)
    result.utility_metric_name = utility_metric
    result.fairness_metric_name = fairness_metric

    return result


def compute_hypervolume(
    pareto_points: List[ConfigurationPoint],
    reference_point: Tuple[float, float] = (0.0, 0.0),
) -> float:
    """Compute the hypervolume indicator for the Pareto frontier.

    The hypervolume measures the area of the objective space dominated
    by the Pareto frontier relative to a reference point.

    Parameters:
        pareto_points: Pareto-optimal points.
        reference_point: Reference point (worst case for both objectives).

    Returns:
        Hypervolume value (float).
    """
    if not pareto_points:
        return 0.0

    # Sort by utility ascending
    sorted_points = sorted(pareto_points, key=lambda p: p.utility)

    hv = 0.0
    prev_utility = reference_point[0]

    for point in sorted_points:
        if point.utility <= reference_point[0] or point.fairness <= reference_point[1]:
            continue
        width = point.utility - prev_utility
        height = point.fairness - reference_point[1]
        hv += width * height
        prev_utility = point.utility

    return hv
