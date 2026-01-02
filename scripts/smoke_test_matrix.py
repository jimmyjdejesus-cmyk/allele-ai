#!/usr/bin/env python3
"""
Smoke Test for Matrix Evaluation System

Quick validation with minimal configuration:
- 1 model
- Baseline + 1 personality
- 1 benchmark
- Limit 10 samples

Usage:
    python scripts/smoke_test_matrix.py
    python scripts/smoke_test_matrix.py --model qwen2.5:0.5b
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.run_matrix_evaluation import MatrixEvaluator
from scripts.analyze_matrix_results import MatrixResultsAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_smoke_test(model: str = None, output_dir: str = "benchmark_results/smoke_test"):
    """Run smoke test with minimal configuration."""
    start_time = time.time()
    
    logger.info("=" * 70)
    logger.info("MATRIX EVALUATION SMOKE TEST")
    logger.info("=" * 70)
    
    # Create evaluator
    evaluator = MatrixEvaluator(
        output_dir=output_dir,
        max_concurrency=1  # Sequential for smoke test
    )
    
    # Use provided model or try to detect one
    models = [model] if model else None
    if not models:
        logger.info("Auto-detecting models...")
        from scripts.detect_model_sizes import detect_ollama_models_in_range
        models = detect_ollama_models_in_range(min_size=0.5, max_size=3.0)
        if not models:
            logger.error("No models detected. Please specify with --model")
            return False
        models = [models[0]]  # Use first detected model
        logger.info(f"Using detected model: {models[0]}")
    
    # Generate minimal matrix
    configs = evaluator.generate_matrix_config(
        models=models,
        personalities=["baseline", "technical_expert"],  # Baseline + 1 personality
        benchmarks=["mmlu"],  # Single benchmark
        min_size=0.5,
        max_size=3.0
    )
    
    logger.info(f"Generated {len(configs)} configurations for smoke test")
    
    # Run evaluation with limit
    logger.info("Running evaluation (limit: 10 samples)...")
    try:
        results = await evaluator.run_matrix_evaluation(
            configs=configs,
            limit=10  # Small limit for quick test
        )
        
        elapsed = time.time() - start_time
        
        # Verify results
        if not results:
            logger.error("No results generated")
            return False
        
        successful = sum(1 for r in results if r.error is None)
        logger.info(f"Results: {successful}/{len(results)} successful")
        
        if successful == 0:
            logger.error("All configurations failed")
            return False
        
        # Verify results file exists
        results_file = Path(output_dir) / "results.json"
        if not results_file.exists():
            logger.error(f"Results file not found: {results_file}")
            return False
        
        logger.info(f"Results file created: {results_file}")
        
        # Quick analysis
        logger.info("Running quick analysis...")
        analyzer = MatrixResultsAnalyzer(results_file)
        stats, best_configs = analyzer.analyze()
        
        if stats:
            logger.info(f"Analyzed {len(stats)} models")
            if best_configs:
                logger.info(f"Best configuration: {best_configs[0].model} + {best_configs[0].personality} ({best_configs[0].average_score:.2f})")
        
        logger.info("=" * 70)
        logger.info(f"SMOKE TEST COMPLETE in {elapsed:.1f}s")
        logger.info("=" * 70)
        
        if elapsed > 300:  # 5 minutes
            logger.warning(f"Smoke test took {elapsed:.1f}s (target: <5 minutes)")
        else:
            logger.info(f"✅ Smoke test completed within time limit ({elapsed:.1f}s < 300s)")
        
        return True
        
    except Exception as e:
        logger.error(f"Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run smoke test for matrix evaluation system"
    )
    parser.add_argument(
        "--model",
        help="Specific model to test (default: auto-detect first available)"
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results/smoke_test",
        help="Output directory for smoke test results"
    )

    args = parser.parse_args()

    success = asyncio.run(run_smoke_test(
        model=args.model,
        output_dir=args.output_dir
    ))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

