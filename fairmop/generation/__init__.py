"""
Module 2 – Image Generation.

Contains the abstract generator base class, the registry system for
pluggable model backends, and built-in generator implementations.
"""

from fairmop.generation.base import BaseGenerator
from fairmop.generation.gpt_image import GPTImageGenerator
from fairmop.generation.registry import GeneratorRegistry

__all__ = [
    "BaseGenerator",
    "GeneratorRegistry",
    "GPTImageGenerator",
]
