"""
Base classes for benchmark implementations.

This module defines the core abstractions for standardized LLM benchmarking.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class BenchmarkResult:
    """Container for benchmark execution results."""

    benchmark_name: str
    score: float
    max_score: float = 100.0
    raw_score: float = field(default=0.0)
    accuracy: float = field(default=0.0)
    pass_at_1: Optional[float] = None
    pass_at_10: Optional[float] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    dataset_size: int = 0
    successful_samples: int = 0

    @property
    def percentage(self) -> float:
        """Return score as percentage."""
        return (self.score / self.max_score) * 100 if self.max_score > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Return success rate as percentage."""
        return (self.successful_samples / self.dataset_size) * 100 if self.dataset_size > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "benchmark_name": self.benchmark_name,
            "score": self.score,
            "max_score": self.max_score,
            "raw_score": self.raw_score,
            "accuracy": self.accuracy,
            "percentage": self.percentage,
            "pass_at_1": self.pass_at_1,
            "pass_at_10": self.pass_at_10,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "error_message": self.error_message,
            "dataset_size": self.dataset_size,
            "successful_samples": self.successful_samples,
            "success_rate": self.success_rate,
        }

    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save results to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> "BenchmarkResult":
        """Load results from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        return cls(**data)


class Benchmark(ABC):
    """Abstract base class for all benchmark implementations."""

    def __init__(self, name: str, description: str, max_score: float = 100.0):
        self.name = name
        self.description = description
        self.max_score = max_score
        self.logger = logging.getLogger(f"benchmarks.{name}")

    @abstractmethod
    async def setup(self) -> None:
        """Setup benchmark (download datasets, prepare environment)."""
        pass

    @abstractmethod
    async def evaluate(self, model: Any, **kwargs) -> BenchmarkResult:
        """Evaluate model and return results."""
        pass

    @abstractmethod
    def get_dataset_size(self) -> int:
        """Return size of benchmark dataset."""
        pass

    def get_prompt_template(self) -> str:
        """Return standard prompt template for this benchmark."""
        return f"Evaluate this {self.name} benchmark query:"

    def cleanup(self) -> None:
        """Cleanup resources after benchmark execution.

        Default is no-op; benchmarks that need to free resources should override
        this method to perform cleanup actions."""
        # Default no-op; provide a debug log so this is not considered an empty method
        self.logger.debug("No cleanup required for this benchmark")

    async def run_benchmark(self, model: Any, **kwargs) -> BenchmarkResult:
        """Run complete benchmark with timing and error handling."""
        start_time = time.time()
        result = BenchmarkResult(
            benchmark_name=self.name,
            score=0.0,
            max_score=self.max_score,
            dataset_size=self.get_dataset_size()
        )

        try:
            self.logger.info(f"Starting {self.name} benchmark evaluation")
            await self.setup()
            result = await self.evaluate(model, **kwargs)

        except Exception as e:
            self.logger.error(f"Error during {self.name} benchmark: {e}")
            result.error_message = str(e)

        finally:
            result.execution_time = time.time() - start_time
            self.cleanup()
            self.logger.info(f"Completed {self.name} benchmark in {result.execution_time:.2f}s")

        return result

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"

    def __repr__(self) -> str:
        return f"Benchmark(name='{self.name}', description='{self.description}')"
