"""
FairMOP Quick Start Example – GPT-Image Backend.

This script demonstrates the minimal code required to run a
FairMOP benchmarking experiment using the OpenAI GPT-Image API.

Prerequisites:
    1. Install FairMOP: ``pip install -e .``
    2. Set your OpenAI API key: ``export OPENAI_API_KEY=sk-...``

Usage:
    python examples/quickstart.py

This will:
    1. Define a small experiment (3 quality levels × 5 images each = 15 images)
    2. Generate images using GPT-Image
    3. Annotate demographics using GPT-4o
    4. Compute CLIP Score & Shannon Entropy
    5. Export results and identify the Pareto frontier
"""

from fairmop import FairMOPPipeline
from fairmop.input_specs import quick_config


def main():
    # ── Option 1: Quick config from a concept template ───────────────────
    config = quick_config(
        concept="nurse",
        model_name="gpt-image",
        grid_params={
            "quality": ["low", "medium", "high"],
        },
        num_images=5,  # 5 images per configuration (small demo)
        metrics=["clip_score", "entropy"],
        output_dir="./fairmop_output/quickstart",
        experiment_name="quickstart_nurse",
        vlm_provider="openai",
        vlm_model="gpt-4o-2024-05-13",
        # api_key="sk-..."  # or set OPENAI_API_KEY env var
    )

    print(config.summary())
    print(f"\nTotal images to generate: {config.total_images()}")

    # ── Run the full pipeline ────────────────────────────────────────────
    pipeline = FairMOPPipeline(config)
    results = pipeline.run()

    # ── Inspect results ──────────────────────────────────────────────────
    print("\n--- Results ---")
    for config_key, data in results["configurations"].items():
        agg = data["aggregates"]
        clip = agg.get("avg_clip_score", "N/A")
        entropy = agg.get("primary_entropy", "N/A")
        print(f"  {config_key}: CLIP={clip}, Entropy={entropy}")


if __name__ == "__main__":
    main()
