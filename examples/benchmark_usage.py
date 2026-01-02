#!/usr/bin/env python3
"""
Example usage of the LM-Eval Benchmarking Pipeline.
Shows how to run benchmarks programmatically.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_lm_eval_results import BenchmarkAnalyzer
from scripts.run_lm_eval_mass import BenchmarkRunner


def main():
    print("🚀 Example: Running LM-Eval Benchmark Pipeline Programmatically")

    # 1. Initialize Runner
    runner = BenchmarkRunner()

    if not runner.check_ollama_running():
        print("❌ Ollama is not running. Please start it first.")
        return

    # 2. Run a Quick Test (Smallest Model)
    print("\n[1] Running quick benchmark on qwen2.5:0.5b...")

    results = runner.run_mass_benchmarks(
        models=["qwen2.5:0.5b"],
        mode="quick",
        custom_limit=5  # Very small limit for example speed
    )

    if not results:
        print("❌ Benchmark failed or produced no results.")
        return

    # 3. Analyze Results
    print("\n[2] Analyzing results...")
    analyzer = BenchmarkAnalyzer()

    # Load and parse
    df = analyzer.parse_metrics(results)

    # Display table
    print("\n--- Benchmark Results ---")
    print(df.to_string(index=False))

    print("\n✅ Example completed successfully!")

if __name__ == "__main__":
    main()

