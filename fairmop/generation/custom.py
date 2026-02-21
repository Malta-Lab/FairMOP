"""
Template for integrating custom T2I model backends into FairMOP.

This module shows how to create a generator for a locally-hosted
model such as Stable Diffusion, FLUX, or any HuggingFace diffusion pipeline.

Usage:
    1. Copy this file or subclass :class:`BaseGenerator`.
    2. Implement the ``generate()`` method.
    3. Register with ``@GeneratorRegistry.register("your-model-name")``.
    4. Reference the name in your YAML config.

Example with Stable Diffusion:

.. code-block:: python

    @GeneratorRegistry.register("stable-diffusion-v1-5")
    class StableDiffusionGenerator(BaseGenerator):
        def setup(self):
            from diffusers import StableDiffusionPipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
            ).to(self.device)

        def generate(self, prompt, seed, **hyperparams):
            import torch
            generator = torch.Generator(self.device).manual_seed(seed)
            result = self.pipe(
                prompt,
                generator=generator,
                guidance_scale=hyperparams.get("guidance_scale", 7.5),
                num_inference_steps=hyperparams.get("num_inference_steps", 50),
            )
            return result.images[0]
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from fairmop.generation.base import BaseGenerator
from fairmop.generation.registry import GeneratorRegistry


# ── Example: Local Diffusion Model ──────────────────────────────────────────


# Uncomment and adapt the following to integrate your own model:

# @GeneratorRegistry.register("stable-diffusion-v1-5")
# class StableDiffusionGenerator(BaseGenerator):
#     """Example generator for Stable Diffusion v1.5."""
#
#     def setup(self):
#         import torch
#         from diffusers import StableDiffusionPipeline
#
#         self.pipe = StableDiffusionPipeline.from_pretrained(
#             "runwayml/stable-diffusion-v1-5",
#             torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
#         ).to(self.device)
#         print(f"[SD1.5] Pipeline loaded on {self.device}")
#
#     def generate(self, prompt: str, seed: int, **hyperparams: Any) -> Image.Image:
#         import torch
#
#         generator = torch.Generator(self.device).manual_seed(seed)
#
#         result = self.pipe(
#             prompt,
#             generator=generator,
#             guidance_scale=hyperparams.get("guidance_scale", 7.5),
#             num_inference_steps=hyperparams.get("num_inference_steps", 50),
#             negative_prompt=hyperparams.get("negative_prompt", None),
#         )
#         return result.images[0]
#
#     def teardown(self):
#         del self.pipe
#         import torch
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()


# @GeneratorRegistry.register("sdxl")
# class SDXLGenerator(BaseGenerator):
#     """Example generator for Stable Diffusion XL 1.0."""
#
#     def setup(self):
#         import torch
#         from diffusers import StableDiffusionXLPipeline
#
#         self.pipe = StableDiffusionXLPipeline.from_pretrained(
#             "stabilityai/stable-diffusion-xl-base-1.0",
#             torch_dtype=torch.float16,
#             variant="fp16",
#         ).to(self.device)
#
#     def generate(self, prompt: str, seed: int, **hyperparams: Any) -> Image.Image:
#         import torch
#
#         generator = torch.Generator(self.device).manual_seed(seed)
#         result = self.pipe(
#             prompt,
#             generator=generator,
#             guidance_scale=hyperparams.get("guidance_scale", 7.5),
#             num_inference_steps=hyperparams.get("num_inference_steps", 50),
#         )
#         return result.images[0]


# @GeneratorRegistry.register("flux-dev")
# class FLUXGenerator(BaseGenerator):
#     """Example generator for FLUX-dev."""
#
#     def setup(self):
#         import torch
#         from diffusers import FluxPipeline
#
#         self.pipe = FluxPipeline.from_pretrained(
#             "black-forest-labs/FLUX.1-dev",
#             torch_dtype=torch.bfloat16,
#         ).to(self.device)
#
#     def generate(self, prompt: str, seed: int, **hyperparams: Any) -> Image.Image:
#         import torch
#
#         generator = torch.Generator(self.device).manual_seed(seed)
#         result = self.pipe(
#             prompt,
#             generator=generator,
#             guidance_scale=hyperparams.get("guidance_scale", 3.5),
#             num_inference_steps=hyperparams.get("num_inference_steps", 50),
#         )
#         return result.images[0]
