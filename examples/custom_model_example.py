"""
FairMOP Example – Custom Model Integration.

Shows how to plug in your own T2I model (e.g., Stable Diffusion, FLUX)
into the FairMOP framework. The key steps are:

    1. Subclass ``BaseGenerator``
    2. Register with ``@GeneratorRegistry.register("your-name")``
    3. Implement ``generate()``
    4. Reference the name in your config

This example uses a dummy generator for demonstration. Replace
the ``DummyGenerator`` with your actual model code.

Usage:
    python examples/custom_model_example.py
"""

import numpy as np
from PIL import Image

from fairmop import ExperimentConfig, FairMOPPipeline
from fairmop.config import HyperparameterGrid
from fairmop.generation import BaseGenerator, GeneratorRegistry

# ── Step 1: Define your custom generator ─────────────────────────────────────


@GeneratorRegistry.register("my-custom-model")
class DummyGenerator(BaseGenerator):
    """A dummy generator that creates colored noise images.

    Replace this with your actual T2I model implementation.
    See ``fairmop/generation/custom.py`` for real examples with
    Stable Diffusion, SDXL, and FLUX.
    """

    def setup(self):
        print(f"[DummyGenerator] Loading model on {self.device}...")
        # In a real implementation:
        # self.pipe = StableDiffusionPipeline.from_pretrained(...)
        # self.pipe.to(self.device)
        print("[DummyGenerator] Model ready!")

    def generate(self, prompt: str, seed: int, **hyperparams) -> Image.Image:
        """Generate a dummy image (random noise).

        In a real implementation, you would call your model here:
            generator = torch.Generator(self.device).manual_seed(seed)
            result = self.pipe(prompt, generator=generator, **hyperparams)
            return result.images[0]
        """
        rng = np.random.RandomState(seed)

        # Use hyperparameters to influence the output
        size = hyperparams.get("resolution", 512)
        guidance = hyperparams.get("guidance_scale", 7.5)

        # Generate a colored noise image (dummy)
        noise = rng.randint(0, 256, (size, size, 3), dtype=np.uint8)

        # Tint the image based on guidance scale (just for demo)
        tint = np.array(
            [
                min(255, int(guidance * 20)),
                min(255, int(100 + seed * 5)),
                min(255, int(150 + guidance * 10)),
            ]
        )
        blended = (noise * 0.7 + tint * 0.3).clip(0, 255).astype(np.uint8)

        return Image.fromarray(blended)

    def teardown(self):
        print("[DummyGenerator] Cleaning up...")
        # del self.pipe
        # torch.cuda.empty_cache()


# ── Step 2: Configure and run the experiment ─────────────────────────────────


def main():
    # Check that our custom generator is registered
    print(f"Available generators: {GeneratorRegistry.available()}")

    config = ExperimentConfig(
        prompt="the face of a nurse",
        experiment_name="custom_model_demo",
        # Reference our custom generator by name
        model_name="my-custom-model",
        model_params={},
        hyperparameter_grid=HyperparameterGrid(
            params={
                "guidance_scale": [3.0, 7.5, 12.0],
                "resolution": [256, 512],
            }
        ),
        num_images_per_config=5,
        seed_start=1,
        # Evaluation settings
        protected_attribute="gender",
        metrics=["clip_score"],  # Only CLIP (no VLM needed for demo)
        output_dir="./fairmop_output/custom_model_demo",
    )

    print(config.summary())

    pipeline = FairMOPPipeline(config)

    # You can also skip evaluation for generation-only runs:
    # results = pipeline.run(skip_evaluation=True)

    results = pipeline.run()

    print("\nDone! Check the output directory for results.")


if __name__ == "__main__":
    main()
