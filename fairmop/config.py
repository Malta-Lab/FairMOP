"""
Experiment configuration for FairMOP.

Defines the :class:`ExperimentConfig` dataclass that holds all parameters
required to run a complete FairMOP benchmarking pipeline, including:
    - Prompt and generation settings
    - Hyperparameter grid
    - Model backend specification
    - Protected attributes and VLM judge settings
    - GPU allocation
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class HyperparameterGrid:
    """Defines a discrete hyperparameter grid for exhaustive evaluation.

    Each key in ``params`` maps a parameter name to a list of values.
    The Cartesian product of all parameter lists defines the full
    configuration space.

    Example::

        grid = HyperparameterGrid(params={
            "guidance_scale": [1.0, 3.5, 7.0],
            "num_inference_steps": [20, 50],
        })
        # produces 6 configurations
    """

    params: Dict[str, List[Any]] = field(default_factory=dict)

    def configurations(self) -> List[Dict[str, Any]]:
        """Return the Cartesian product of all parameter lists."""
        if not self.params:
            return [{}]
        keys = list(self.params.keys())
        values = list(self.params.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def __len__(self) -> int:
        configs = self.configurations()
        return len(configs)


@dataclass
class ExperimentConfig:
    """Complete specification of a FairMOP experiment.

    Attributes:
        prompt: Text prompt for image generation.
        model_name: Name/identifier of the T2I model backend.
        model_params: Extra keyword arguments forwarded to the generator.
        hyperparameter_grid: Grid of hyperparameters to sweep.
        num_images_per_config: Number of images per grid configuration.
        seed_start: Starting random seed (seeds are sequential).
        protected_attribute: Demographic axis to evaluate (gender, ethnicity, age).
        vlm_provider: API provider for the VLM judge (openai | gemini).
        vlm_model: Specific VLM model name.
        api_key: API key for the VLM provider (or use env variable).
        gpu_index: CUDA device index (None → CPU).
        output_dir: Directory where results are saved.
        experiment_name: Human-readable experiment label.
        metrics: List of metrics to compute.
        reference_images_dir: Path to real images for FID / PRDC.
    """

    # ── Generation ──────────────────────────────────────────────────────
    prompt: str = "the face of a nurse"
    model_name: str = "gpt-image"
    model_params: Dict[str, Any] = field(default_factory=dict)
    hyperparameter_grid: HyperparameterGrid = field(default_factory=HyperparameterGrid)
    num_images_per_config: int = 100
    seed_start: int = 1

    # ── Evaluation ──────────────────────────────────────────────────────
    protected_attribute: str = "gender"
    vlm_provider: str = "openai"
    vlm_model: str = "gpt-4o-2024-05-13"
    api_key: Optional[str] = None
    metrics: List[str] = field(default_factory=lambda: ["clip_score", "entropy", "kl"])
    reference_images_dir: Optional[str] = None

    # ── Infrastructure ──────────────────────────────────────────────────
    gpu_index: Optional[int] = None
    output_dir: str = "./fairmop_output"
    experiment_name: str = "fairmop_experiment"

    # ── Naming ──────────────────────────────────────────────────────────
    image_filename_template: str = "{prompt}_{params}_seed{seed}.png"

    def __post_init__(self):
        if self.api_key is None:
            if self.vlm_provider == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY")
            elif self.vlm_provider == "gemini":
                self.api_key = os.environ.get("GEMINI_API_KEY")

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from a YAML file.

        Parameters:
            path: Path to the YAML configuration file.

        Returns:
            A fully initialized :class:`ExperimentConfig`.
        """
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        grid_data = raw.pop("hyperparameter_grid", {})
        grid = HyperparameterGrid(params=grid_data)
        return cls(hyperparameter_grid=grid, **raw)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Create configuration from a plain dictionary."""
        grid_data = d.pop("hyperparameter_grid", {})
        grid = HyperparameterGrid(params=grid_data)
        return cls(hyperparameter_grid=grid, **d)

    def to_yaml(self, path: str) -> None:
        """Serialize the config to a YAML file."""
        d = self.__dict__.copy()
        d["hyperparameter_grid"] = self.hyperparameter_grid.params
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(d, fh, default_flow_style=False, sort_keys=False)

    def total_images(self) -> int:
        """Total number of images that will be generated."""
        return len(self.hyperparameter_grid) * self.num_images_per_config

    def summary(self) -> str:
        """Human-readable summary of the experiment setup."""
        n_configs = len(self.hyperparameter_grid)
        total = self.total_images()
        lines = [
            f"Experiment : {self.experiment_name}",
            f'Prompt     : "{self.prompt}"',
            f"Model      : {self.model_name}",
            f"Grid       : {n_configs} configurations",
            f"Images/cfg : {self.num_images_per_config}",
            f"Total imgs : {total}",
            f"Metrics    : {', '.join(self.metrics)}",
            f"Attribute  : {self.protected_attribute}",
            f"VLM Judge  : {self.vlm_provider} / {self.vlm_model}",
            f"GPU        : {self.gpu_index if self.gpu_index is not None else 'CPU'}",
            f"Output     : {self.output_dir}",
        ]
        return "\n".join(lines)
