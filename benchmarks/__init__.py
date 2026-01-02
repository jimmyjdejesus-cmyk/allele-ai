"""
Benchmark Suite for Phylogenic-AI-Agents

This module provides standardized LLM benchmarking capabilities
including MMLU, HellaSwag, HumanEval, GSM8K, and other key benchmarks.
"""

from .base import Benchmark, BenchmarkResult
from .registry import BenchmarkRegistry

__all__ = ["Benchmark", "BenchmarkResult", "BenchmarkRegistry"]
