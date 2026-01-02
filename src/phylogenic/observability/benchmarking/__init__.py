# Copyright (C) 2025 Phylogenic AI Labs & Jimmy De Jesus
#
# This file is part of Allele.
#
# Allele is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Allele is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY, without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Allele.  If not, see <https://www.gnu.org/licenses/>.
#
# =============================================================================
# COMMERCIAL LICENSE:
# If you wish to use this software in a proprietary/closed-source application
# without releasing your source code, you must purchase a Commercial License
# from: https://gumroad.com/l/[YOUR_LINK]
# =============================================================================

"""Matrix Benchmarking System for Allele.

This module provides comprehensive matrix benchmarking capabilities for testing
performance across different parameter combinations.

Author: Bravetto AI Systems
Version: 1.0.0
"""

from typing import Any

try:
    # Optional benchmarking components; import if available at runtime
    import importlib

    _mod = importlib.import_module(__name__ + ".comparator")
    BenchmarkComparator = _mod.BenchmarkComparator
except Exception:  # pragma: no cover - optional
    BenchmarkComparator = Any

from .config import MatrixBenchmarkSettings

try:
    import importlib

    _mod = importlib.import_module(__name__ + ".integration")
    PytestBenchmarkIntegration = _mod.PytestBenchmarkIntegration
except Exception:  # pragma: no cover - optional
    PytestBenchmarkIntegration = Any

try:
    import importlib

    _mod = importlib.import_module(__name__ + ".matrix_runner")
    MatrixBenchmarkRunner = _mod.MatrixBenchmarkRunner
except Exception:  # pragma: no cover - optional
    MatrixBenchmarkRunner = Any

try:
    import importlib

    _mod = importlib.import_module(__name__ + ".profiler")
    PerformanceProfiler = _mod.PerformanceProfiler
except Exception:  # pragma: no cover - optional
    PerformanceProfiler = Any
from .types import BenchmarkConfig, BenchmarkResult, PerformanceProfile

__all__ = [
    "MatrixBenchmarkRunner",
    "PerformanceProfiler",
    "BenchmarkComparator",
    "PytestBenchmarkIntegration",
    "MatrixBenchmarkSettings",
    "BenchmarkConfig",
    "BenchmarkResult",
    "PerformanceProfile",
]
