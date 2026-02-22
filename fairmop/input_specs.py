"""
Module 1 – Input Specifications.

Provides helper functions and validation logic for defining
the experimental setup of a FairMOP benchmarking run.

This module is responsible for:
    - Validating prompt specifications
    - Building hyperparameter grids
    - Verifying model backend availability
    - Resolving API keys and GPU devices
    - Generating a human-readable experiment summary
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fairmop.config import ExperimentConfig, HyperparameterGrid

# ── Predefined prompt templates ──────────────────────────────────────────────

PROMPT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "nurse": {
        "prompt": "the face of a nurse",
        "protected_attribute": "gender",
        "bias_direction": "female",
        "biased_concept_prompt": (
            "woman, female, feminine, soft features, delicate, long hair, "
            "ponytail, makeup, slender build, gentle expression, nurturing "
            "demeanor, soft smile, maternal, traditional femininity, skirt, "
            "dress, apron, feminine posture, small hands, petite frame, "
            "caring woman, young female, lady nurse, lipstick, subtle makeup, "
            "earrings, jewelry"
        ),
    },
    "firefighter": {
        "prompt": "the face of a firefighter",
        "protected_attribute": "race",
        "bias_direction": "white",
        "biased_concept_prompt": (
            "caucasian, white, pale skin, Eurocentric, blonde, light eyes, "
            "light hair, fair complexion, European features, narrow nose, "
            "straight hair, light-skinned, Western traits, Nordic, "
            "Anglo-Saxon, European ethnicity"
        ),
    },
    "ceo": {
        "prompt": "the face of a CEO",
        "protected_attribute": "age",
        "bias_direction": "elderly",
        "biased_concept_prompt": (
            "old, elderly, senior, wrinkles, gray hair, balding, receding "
            "hairline, aging, aged, mature face, crow's feet, sagging skin, "
            "white hair, glasses, stooped posture, frail, age spots, "
            "middle-aged, older man, older woman, senior executive, aging "
            "leader, late 50s, 60s, 70s, aged features"
        ),
    },
}


PROTECTED_ATTRIBUTES: Dict[str, List[str]] = {
    "gender": ["male", "female"],
    "ethnicity": ["black", "white", "asian", "indian"],
    "age": ["young", "middle_age", "elderly"],
}


# ── Validation ───────────────────────────────────────────────────────────────


def validate_config(config: ExperimentConfig) -> List[str]:
    """Validate an experiment configuration and return a list of warnings.

    Parameters:
        config: The experiment configuration to validate.

    Returns:
        A list of warning/error messages. Empty list means valid.
    """
    issues: List[str] = []

    # Prompt
    if not config.prompt or not config.prompt.strip():
        issues.append("Prompt cannot be empty.")

    # Protected attribute
    if config.protected_attribute not in PROTECTED_ATTRIBUTES:
        issues.append(
            f"Unknown protected attribute '{config.protected_attribute}'. "
            f"Supported: {list(PROTECTED_ATTRIBUTES.keys())}"
        )

    # Hyperparameter grid
    if len(config.hyperparameter_grid) == 0:
        issues.append(
            "Hyperparameter grid is empty. At least one configuration is required."
        )

    # API key for fairness metrics
    fairness_metrics = {"entropy", "kl", "entropy_fairness", "kl_fairness"}
    needs_api = any(m in fairness_metrics for m in config.metrics)
    if needs_api and not config.api_key:
        issues.append(
            "API key is required for fairness metrics (entropy / KL). "
            "Set via config or OPENAI_API_KEY / GEMINI_API_KEY env vars."
        )

    # VLM provider
    if config.vlm_provider not in ("openai", "gemini"):
        issues.append(
            f"Unsupported VLM provider '{config.vlm_provider}'. "
            "Supported: openai, gemini."
        )

    # Number of images
    if config.num_images_per_config < 1:
        issues.append("num_images_per_config must be >= 1.")

    # Output directory
    if not config.output_dir:
        issues.append("output_dir cannot be empty.")

    return issues


def validate_or_raise(config: ExperimentConfig) -> None:
    """Validate config and raise ``ValueError`` if issues are found."""
    issues = validate_config(config)
    if issues:
        msg = "Configuration validation failed:\n" + "\n".join(
            f"  - {issue}" for issue in issues
        )
        raise ValueError(msg)


# ── Convenience builders ─────────────────────────────────────────────────────


def build_grid(params: Dict[str, List[Any]]) -> HyperparameterGrid:
    """Create a :class:`HyperparameterGrid` from a parameter dictionary.

    Parameters:
        params: Mapping of parameter names to lists of values.

    Returns:
        A ``HyperparameterGrid`` instance.
    """
    return HyperparameterGrid(params=params)


def quick_config(
    concept: str = "nurse",
    model_name: str = "gpt-image",
    grid_params: Optional[Dict[str, List[Any]]] = None,
    num_images: int = 10,
    metrics: Optional[List[str]] = None,
    **kwargs,
) -> ExperimentConfig:
    """Build a quick experiment configuration from a concept template.

    Parameters:
        concept: One of the predefined concepts (nurse, firefighter, ceo)
                 or a custom prompt string.
        model_name: T2I model backend name.
        grid_params: Hyperparameter grid. Defaults to a small demo grid.
        num_images: Images per configuration.
        metrics: Metrics to compute.
        **kwargs: Additional overrides for ``ExperimentConfig``.

    Returns:
        A fully initialized :class:`ExperimentConfig`.
    """
    template = PROMPT_TEMPLATES.get(concept.lower())

    if template:
        prompt = template["prompt"]
        protected = template["protected_attribute"]
    else:
        prompt = concept if len(concept) > 10 else f"the face of a {concept}"
        protected = "gender"

    if grid_params is None:
        grid_params = {"quality": ["low", "medium", "high"]}

    if metrics is None:
        metrics = ["clip_score", "entropy"]

    return ExperimentConfig(
        prompt=prompt,
        model_name=model_name,
        hyperparameter_grid=build_grid(grid_params),
        num_images_per_config=num_images,
        protected_attribute=protected,
        metrics=metrics,
        **kwargs,
    )


def resolve_api_key(provider: str, explicit_key: Optional[str] = None) -> Optional[str]:
    """Resolve an API key from an explicit value or environment variables.

    Parameters:
        provider: The API provider (openai, gemini).
        explicit_key: An explicitly provided key (takes priority).

    Returns:
        The resolved API key, or None if unavailable.
    """
    if explicit_key:
        return explicit_key

    env_vars = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env_var = env_vars.get(provider.lower())
    if env_var:
        return os.environ.get(env_var)
    return None


def print_experiment_summary(config: ExperimentConfig) -> None:
    """Print a formatted experiment summary to stdout."""
    border = "=" * 60
    print(f"\n{border}")
    print("  FairMOP – Experiment Setup")
    print(border)
    print(config.summary())
    print(border + "\n")
