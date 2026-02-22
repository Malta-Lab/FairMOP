"""
FairMOP Example – GPT-Image Full Experiment.

Demonstrates a complete benchmarking experiment with the OpenAI
GPT-Image API, including multiple quality levels and image sizes.

This example is more comprehensive than the quickstart and shows
how to configure a realistic experiment setup.

Usage:
    python examples/gpt_image_example.py
"""

from fairmop import ExperimentConfig, FairMOPPipeline
from fairmop.config import HyperparameterGrid


def main():
    config = ExperimentConfig(
        # ── Prompt ───────────────────────────────────────────────────
        prompt="the face of a nurse",
        experiment_name="gpt_image_nurse_benchmark",
        # ── Model ────────────────────────────────────────────────────
        model_name="gpt-image",
        model_params={
            "openai_model": "gpt-image-1",
            "rate_limit_delay": 1.0,
        },
        # ── Hyperparameter Grid ──────────────────────────────────────
        # Each combination defines one configuration
        hyperparameter_grid=HyperparameterGrid(
            params={
                "quality": ["low", "medium", "high"],
                "size": ["1024x1024", "1536x1024"],
            }
        ),
        # ── Generation ───────────────────────────────────────────────
        num_images_per_config=20,  # 20 images per configuration
        seed_start=1,
        # ── Evaluation ───────────────────────────────────────────────
        protected_attribute="gender",
        vlm_provider="openai",
        vlm_model="gpt-4o-2024-05-13",
        metrics=["clip_score", "entropy", "kl"],
        # ── Output ───────────────────────────────────────────────────
        output_dir="./fairmop_output/gpt_image_experiment",
    )

    # Print experiment overview
    print(config.summary())
    n = len(config.hyperparameter_grid)
    print(f"\nConfigurations: {n}")
    print(f"Total images: {config.total_images()}")
    print("Grid:")
    for i, cfg in enumerate(config.hyperparameter_grid.configurations()):
        print(f"  [{i + 1}] {cfg}")

    # Run pipeline
    pipeline = FairMOPPipeline(config)
    results = pipeline.run()

    # Print Pareto summary
    from fairmop.output.pareto import build_pareto_from_results

    pareto = build_pareto_from_results(
        results,
        utility_metric="avg_clip_score",
        fairness_metric="primary_entropy",
    )
    print(f"\n{pareto.summary()}")


if __name__ == "__main__":
    main()
