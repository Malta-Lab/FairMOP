"""
Generator registry – plugin system for T2I model backends.

Register your custom generator with::

    from fairmop.generation import GeneratorRegistry, BaseGenerator

    @GeneratorRegistry.register("my-custom-model")
    class MyGenerator(BaseGenerator):
        def generate(self, prompt, seed, **hyperparams):
            ...
"""

from __future__ import annotations

from typing import Any, Dict, Type

from fairmop.generation.base import BaseGenerator


class GeneratorRegistry:
    """Central registry for T2I generator backends.

    Provides decorator-based registration and factory instantiation.
    """

    _registry: Dict[str, Type[BaseGenerator]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a generator class under a given name.

        Parameters:
            name: Identifier for the generator (e.g., ``"stable-diffusion"``).

        Usage::

            @GeneratorRegistry.register("my-model")
            class MyModelGenerator(BaseGenerator):
                ...
        """

        def decorator(generator_cls: Type[BaseGenerator]):
            if not issubclass(generator_cls, BaseGenerator):
                raise TypeError(f"{generator_cls.__name__} must subclass BaseGenerator")
            cls._registry[name.lower()] = generator_cls
            return generator_cls

        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> BaseGenerator:
        """Instantiate a registered generator by name.

        Parameters:
            name: Registered name of the generator.
            device: PyTorch device string.
            **kwargs: Additional keyword arguments forwarded to the constructor.

        Returns:
            An instance of the registered generator.

        Raises:
            KeyError: If the name is not registered.
        """
        name_lower = name.lower()
        if name_lower not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise KeyError(
                f"Generator '{name}' not found. "
                f"Available: [{available}]. "
                f"Register custom generators with "
                f"@GeneratorRegistry.register('{name}')"
            )
        generator_cls = cls._registry[name_lower]
        return generator_cls(model_name=name, device=device, **kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """Return a sorted list of all registered generator names."""
        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check whether a generator name is registered."""
        return name.lower() in cls._registry
