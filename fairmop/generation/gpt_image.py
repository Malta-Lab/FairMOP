"""
GPT-Image generator backend.

Uses the OpenAI ``gpt-image-1`` model via the Images API
to generate images from text prompts.

Reference: https://platform.openai.com/docs/guides/images

This generator serves as the default built-in example for FairMOP.
It demonstrates how to integrate an API-based T2I model into the framework.
"""

from __future__ import annotations

import base64
import io
import os
import time
from typing import Any, Optional

from PIL import Image

from fairmop.generation.base import BaseGenerator
from fairmop.generation.registry import GeneratorRegistry


@GeneratorRegistry.register("gpt-image")
class GPTImageGenerator(BaseGenerator):
    """Generator backend for OpenAI's GPT-Image model.

    Parameters:
        model_name: Model identifier (default: ``"gpt-image"``).
        device: Ignored for API-based generators.
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
        openai_model: Actual OpenAI model name (``"gpt-image-1"``).
        size: Image dimensions (e.g., ``"1024x1024"``).
        quality: Image quality (``"low"``, ``"medium"``, ``"high"``).
        rate_limit_delay: Delay in seconds between API calls.
    """

    def __init__(
        self,
        model_name: str = "gpt-image",
        device: str = "cpu",
        api_key: Optional[str] = None,
        openai_model: str = "gpt-image-1",
        size: str = "1024x1024",
        quality: str = "medium",
        rate_limit_delay: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, device=device, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.openai_model = openai_model
        self.size = size
        self.quality = quality
        self.rate_limit_delay = rate_limit_delay
        self._client = None

    def setup(self) -> None:
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for GPTImageGenerator. "
                "Install with: pip install openai"
            )

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY env var "
                "or pass api_key to the generator."
            )

        self._client = OpenAI(api_key=self.api_key)
        print(f"[GPTImageGenerator] Initialized with model: {self.openai_model}")

    def generate(
        self,
        prompt: str,
        seed: int,
        **hyperparams: Any,
    ) -> Image.Image:
        """Generate an image using the OpenAI Images API.

        Parameters:
            prompt: Text prompt for generation.
            seed: Random seed (appended to prompt for variation since the
                  API doesn't support explicit seeds).
            **hyperparams: Override ``quality`` and ``size`` per call.

        Returns:
            A PIL Image object.
        """
        if self._client is None:
            self.setup()

        quality = hyperparams.get("quality", self.quality)
        size = hyperparams.get("size", self.size)

        # The GPT-Image API doesn't support explicit seeds, so we append
        # a seed marker to the prompt for slight variation
        seeded_prompt = f"{prompt} [variation {seed}]"

        try:
            # Rate limiting
            time.sleep(self.rate_limit_delay)

            result = self._client.images.generate(
                model=self.openai_model,
                prompt=seeded_prompt,
                size=size,
                quality=quality,
                n=1,
                output_format="png",
            )

            # Handle base64-encoded response
            if hasattr(result.data[0], "b64_json") and result.data[0].b64_json:
                image_bytes = base64.b64decode(result.data[0].b64_json)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            elif hasattr(result.data[0], "url") and result.data[0].url:
                import requests

                response = requests.get(result.data[0].url, timeout=30)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
            else:
                raise ValueError("No image data received from API")

            return image

        except Exception as e:
            print(f"[GPTImageGenerator] Error generating image (seed={seed}): {e}")
            raise

    def teardown(self) -> None:
        """Clean up the client."""
        self._client = None
