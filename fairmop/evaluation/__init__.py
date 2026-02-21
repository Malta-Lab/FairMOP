"""
Module 3 – Evaluation.

Provides fairness and utility metric computation, as well as
VLM-based demographic annotation for generated images.
"""

from fairmop.evaluation.fairness import (
    compute_fairness_metrics,
    kl_divergence_from_uniform,
    normalized_shannon_entropy,
)
from fairmop.evaluation.utility import (
    compute_clip_score,
    compute_fid,
    compute_prdc_precision,
)
from fairmop.evaluation.vlm_judge import VLMJudge

__all__ = [
    "VLMJudge",
    "normalized_shannon_entropy",
    "kl_divergence_from_uniform",
    "compute_fairness_metrics",
    "compute_clip_score",
    "compute_fid",
    "compute_prdc_precision",
]
