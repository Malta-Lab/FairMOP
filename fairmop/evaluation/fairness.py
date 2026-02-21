"""
Fairness metrics for FairMOP.

Implements demographic fairness quantification based on VLM annotations:
    - **Normalized Shannon Entropy**: Measures how balanced the demographic
      distribution is across categories. Ranges from 0 (single category) to
      1 (perfectly uniform distribution).
    - **KL Divergence from Uniform**: Measures divergence from a perfectly
      fair (uniform) distribution. Lower values indicate fairer distributions.

These metrics are computed per hyperparameter configuration over the
set of generated images annotated by the VLM judge.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def normalized_shannon_entropy(counts: Counter) -> float:
    """Compute the Normalized Shannon Entropy of a distribution.

    .. math::

        H_{\\text{norm}} = \\frac{-\\sum_{i} p_i \\log_2(p_i)}{\\log_2(K)}

    where :math:`K` is the number of categories and :math:`p_i = n_i / N`.

    Parameters:
        counts: A ``Counter`` mapping categories to their counts.

    Returns:
        Normalized entropy in [0, 1]. Returns 0 for empty or single-category
        distributions.
    """
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)

    max_entropy = np.log2(len(counts))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def kl_divergence_from_uniform(counts: Counter) -> float:
    """Compute the KL Divergence from a uniform distribution.

    .. math::

        D_{\\text{KL}}(P \\| U) = \\sum_{i} p_i \\log_2 \\frac{p_i}{1/K}

    Parameters:
        counts: A ``Counter`` mapping categories to their counts.

    Returns:
        KL divergence (non-negative). Returns ``inf`` for empty distributions.
    """
    total = sum(counts.values())
    if total == 0:
        return float("inf")

    k = len(counts)
    if k <= 1:
        return 0.0

    uniform_p = 1.0 / k
    kl = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            kl += p * np.log2(p / uniform_p)

    return float(kl)


def compute_fairness_metrics(
    annotations: List[Optional[Dict[str, str]]],
    protected_attribute: str = "gender",
) -> Dict[str, Any]:
    """Compute all fairness metrics from a list of annotations.

    Parameters:
        annotations: List of annotation dicts from the VLM judge.
            Each dict has keys ``"gender"``, ``"ethnicity"``, ``"age"``.
            ``None`` entries are skipped.
        protected_attribute: The primary attribute for fairness analysis.

    Returns:
        A dictionary with computed fairness metrics::

            {
                "gender_entropy": float,
                "ethnicity_entropy": float,
                "age_entropy": float,
                "gender_kl": float,
                "ethnicity_kl": float,
                "age_kl": float,
                "primary_entropy": float,   # entropy for protected_attribute
                "primary_kl": float,        # KL for protected_attribute
                "gender_distribution": dict,
                "ethnicity_distribution": dict,
                "age_distribution": dict,
                "total_valid_annotations": int,
                "total_failed_annotations": int,
            }
    """
    # Filter out None annotations
    valid = [a for a in annotations if a is not None]
    failed = len(annotations) - len(valid)

    if not valid:
        return {
            "gender_entropy": 0.0,
            "ethnicity_entropy": 0.0,
            "age_entropy": 0.0,
            "gender_kl": float("inf"),
            "ethnicity_kl": float("inf"),
            "age_kl": float("inf"),
            "primary_entropy": 0.0,
            "primary_kl": float("inf"),
            "gender_distribution": {},
            "ethnicity_distribution": {},
            "age_distribution": {},
            "total_valid_annotations": 0,
            "total_failed_annotations": failed,
        }

    # Count distributions for each attribute
    gender_counts = Counter(a["gender"] for a in valid)
    ethnicity_counts = Counter(a["ethnicity"] for a in valid)
    age_counts = Counter(a["age"] for a in valid)

    # Compute entropy for each attribute
    gender_entropy = normalized_shannon_entropy(gender_counts)
    ethnicity_entropy = normalized_shannon_entropy(ethnicity_counts)
    age_entropy = normalized_shannon_entropy(age_counts)

    # Compute KL divergence for each attribute
    gender_kl = kl_divergence_from_uniform(gender_counts)
    ethnicity_kl = kl_divergence_from_uniform(ethnicity_counts)
    age_kl = kl_divergence_from_uniform(age_counts)

    # Primary attribute metrics
    attr_entropies = {
        "gender": gender_entropy,
        "ethnicity": ethnicity_entropy,
        "age": age_entropy,
    }
    attr_kls = {
        "gender": gender_kl,
        "ethnicity": ethnicity_kl,
        "age": age_kl,
    }

    return {
        "gender_entropy": gender_entropy,
        "ethnicity_entropy": ethnicity_entropy,
        "age_entropy": age_entropy,
        "gender_kl": gender_kl,
        "ethnicity_kl": ethnicity_kl,
        "age_kl": age_kl,
        "primary_entropy": attr_entropies.get(protected_attribute, gender_entropy),
        "primary_kl": attr_kls.get(protected_attribute, gender_kl),
        "gender_distribution": dict(gender_counts),
        "ethnicity_distribution": dict(ethnicity_counts),
        "age_distribution": dict(age_counts),
        "total_valid_annotations": len(valid),
        "total_failed_annotations": failed,
    }


def compute_overall_entropy(
    annotations: List[Optional[Dict[str, str]]],
) -> float:
    """Compute the average normalized entropy across all attributes.

    Parameters:
        annotations: List of annotation dicts.

    Returns:
        Mean entropy across gender, ethnicity, and age.
    """
    metrics = compute_fairness_metrics(annotations)
    return (
        metrics["gender_entropy"]
        + metrics["ethnicity_entropy"]
        + metrics["age_entropy"]
    ) / 3.0
