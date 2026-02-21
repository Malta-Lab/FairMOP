"""
Abstract base class for all T2I generator backends.

Any generator integrated into FairMOP must subclass :class:`BaseGenerator`
and implement the :meth:`generate` method.

The framework calls ``generate()`` once per (configuration, seed) pair
and expects a PIL Image in return.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from PIL import Image


class BaseGenerator(ABC):
    """Abstract base class for Text-to-Image generators.

    Subclass this to integrate any T2I model (local or API-based)
    into the FairMOP pipeline.

    Parameters:
        model_name: Identifier for the model (e.g., ``"stable-diffusion-v1-5"``).
        device: PyTorch device string (e.g., ``"cuda:0"`` or ``"cpu"``).
        **kwargs: Additional model-specific parameters.
    """

    def __init__(
        self,
        model_name: str = "",
        device: str = "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.extra_params = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: str,
        seed: int,
        **hyperparams: Any,
    ) -> Image.Image:
        """Generate a single image from a text prompt.

        Parameters:
            prompt: The text prompt to use for generation.
            seed: Random seed for reproducibility.
            **hyperparams: Hyperparameters from the grid (e.g.,
                           ``guidance_scale=7.0``, ``num_inference_steps=50``).

        Returns:
            A PIL Image object.
        """
        ...

    def generate_batch(
        self,
        prompt: str,
        seeds: list[int],
        **hyperparams: Any,
    ) -> list[Image.Image]:
        """Generate a batch of images (default: sequential calls).

        Override for models that support native batching.

        Parameters:
            prompt: The text prompt.
            seeds: List of random seeds.
            **hyperparams: Hyperparameters from the grid.

        Returns:
            List of PIL Image objects.
        """
        return [self.generate(prompt, seed, **hyperparams) for seed in seeds]

    def save_image(
        self,
        image: Image.Image,
        output_dir: str,
        prompt_slug: str,
        params_slug: str,
        seed: int,
        extension: str = "png",
    ) -> str:
        """Save an image following the FairMOP naming convention.

        Filename format: ``{prompt}_{params}_seed{seed}.{ext}``

        Parameters:
            image: The generated PIL Image.
            output_dir: Directory to save the image.
            prompt_slug: Sanitized prompt string for the filename.
            params_slug: Encoded hyperparameters for the filename.
            seed: Random seed used for generation.
            extension: File extension (default: png).

        Returns:
            The absolute path to the saved image.
        """
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{prompt_slug}_{params_slug}_seed{seed}.{extension}"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        return filepath

    @staticmethod
    def slugify_prompt(prompt: str) -> str:
        """Convert a prompt string to a filename-safe slug.

        Example:
            ``"the face of a nurse"`` → ``"nurse"``
        """
        prompt_lower = prompt.lower().strip()
        # Extract concept from "the face of a X" pattern
        if "the face of a " in prompt_lower:
            return prompt_lower.replace("the face of a ", "").replace(" ", "_")
        if "the face of an " in prompt_lower:
            return prompt_lower.replace("the face of an ", "").replace(" ", "_")
        # General slugification
        slug = prompt_lower.replace(" ", "_")
        slug = "".join(c for c in slug if c.isalnum() or c == "_")
        return slug[:50]  # Truncate to reasonable length

    @staticmethod
    def slugify_params(hyperparams: Dict[str, Any]) -> str:
        """Encode hyperparameters into a filename-safe string.

        Example:
            ``{"guidance_scale": 7.0, "steps": 50}`` →
            ``"guidance_scale7.0_steps50"``
        """
        if not hyperparams:
            return "default"
        parts = []
        for key, value in sorted(hyperparams.items()):
            # Shorten common parameter names
            short_key = key.replace("guidance_scale", "cfg")
            short_key = short_key.replace("num_inference_steps", "steps")
            short_key = short_key.replace("edit_guidance_scale", "editguidance")
            parts.append(f"{short_key}{value}")
        return "_".join(parts)

    def setup(self) -> None:
        """Optional setup hook called before generation begins.

        Override to load models, initialize pipelines, etc.
        """
        pass

    def teardown(self) -> None:
        """Optional teardown hook called after generation completes.

        Override to free resources, unload models, etc.
        """
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', device='{self.device}')"
        )
