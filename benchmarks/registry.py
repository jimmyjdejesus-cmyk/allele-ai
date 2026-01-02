"""
Benchmark registry for managing and discovering available benchmarks.

This module provides a centralized registry for all benchmark implementations.
"""

import logging
from typing import Dict, List, Optional, Type

from .base import Benchmark


class BenchmarkRegistry:
    """Registry for managing benchmark implementations."""

    _registry: Dict[str, Type[Benchmark]] = {}
    _logger = logging.getLogger("benchmarks.registry")

    @classmethod
    def register(cls, name: str):
        """Decorator to register a benchmark class."""
        def decorator(benchmark_class: Type[Benchmark]) -> Type[Benchmark]:
            cls._registry[name] = benchmark_class
            cls._logger.info(f"Registered benchmark: {name}")
            return benchmark_class
        return decorator

    @classmethod
    def get_benchmark(cls, name: str) -> Optional[Type[Benchmark]]:
        """Get benchmark class by name."""
        return cls._registry.get(name)

    @classmethod
    def get_all_benchmarks(cls) -> Dict[str, Type[Benchmark]]:
        """Get all registered benchmarks."""
        return cls._registry.copy()

    @classmethod
    def list_benchmarks(cls) -> List[str]:
        """Get list of all registered benchmark names."""
        return list(cls._registry.keys())

    @classmethod
    def create_instance(cls, name: str, **kwargs) -> Optional[Benchmark]:
        """Create an instance of a benchmark by name."""
        benchmark_class = cls.get_benchmark(name)
        if benchmark_class:
            return benchmark_class(**kwargs)
        return None

    @classmethod
    def get_benchmark_info(cls) -> Dict[str, Dict[str, str]]:
        """Get information about all registered benchmarks."""
        info = {}
        for name, klass in cls._registry.items():
            info[name] = {
                "class_name": klass.__name__,
                "module": klass.__module__,
                "description": getattr(klass, '__doc__', 'No description available')
            }
        return info


# Global registry instance
registry = BenchmarkRegistry()

# Convenience functions
def register_benchmark(name: str):
    """Register a benchmark class with the global registry."""
    return registry.register(name)

def get_benchmark_class(name: str) -> Optional[Type[Benchmark]]:
    """Get benchmark class by name."""
    return registry.get_benchmark(name)

def list_available_benchmarks() -> List[str]:
    """Get list of all available benchmark names."""
    return registry.list_benchmarks()

def create_benchmark_instance(name: str, **kwargs) -> Optional[Benchmark]:
    """Create benchmark instance by name."""
    return registry.create_instance(name, **kwargs)
