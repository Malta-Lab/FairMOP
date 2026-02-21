"""
FairMOP: Benchmarking Fairness-Utility Trade-offs in Text-to-Image Models
via Pareto Frontiers.

A modular, model-agnostic framework for evaluating fairness and utility
in text-to-image (T2I) generative models. FairMOP treats the evaluation
as a multi-objective optimization problem (MOOP), identifying
Pareto-optimal configurations in the fairness-utility space.

Modules:
    - input_specs: Define experimental setups (prompts, hyperparameters, models)
    - generation: Generate images using pluggable T2I model backends
    - evaluation: Compute fairness (entropy, KL) and utility (CLIP, FID, PRDC) metrics
    - output: Visualize Pareto frontiers and export results

Quick Start:
    >>> from fairmop import FairMOPPipeline, ExperimentConfig
    >>> config = ExperimentConfig.from_yaml("experiment.yaml")
    >>> pipeline = FairMOPPipeline(config)
    >>> results = pipeline.run()
"""

__version__ = "0.1.0"
__author__ = "FairMOP Team"

from fairmop.config import ExperimentConfig
from fairmop.generation.registry import GeneratorRegistry
from fairmop.pipeline import FairMOPPipeline

__all__ = [
    "ExperimentConfig",
    "FairMOPPipeline",
    "GeneratorRegistry",
    "__version__",
]
