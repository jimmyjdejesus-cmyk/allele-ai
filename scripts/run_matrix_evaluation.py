#!/usr/bin/env python3
"""
Multi-Model Personality Matrix Evaluation Runner

Tests all combinations of:
- Models: Auto-detected Ollama models (0.5b-3b range)
- Personalities: Baseline, 5 base personalities, COT prompts
- Benchmarks: MMLU, HellaSwag, GSM8K, ARC-Easy, TruthfulQA

Usage:
    python scripts/run_matrix_evaluation.py --mode quick --limit 10
    python scripts/run_matrix_evaluation.py --resume checkpoint.json
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
from scripts.detect_model_sizes import detect_ollama_models_in_range
from scripts.run_lm_eval_mass import BenchmarkRunner, BENCHMARK_TASKS
from scripts.direct_ollama_benchmark import DirectOllamaBenchmarkRunner

# Import personality archetypes - need to define locally to avoid import issues
PERSONALITY_ARCHETYPES = {
    "baseline": None,  # No genome, raw model
    "technical_expert": {
        "empathy": 0.2,
        "technical_knowledge": 0.99,
        "creativity": 0.3,
        "conciseness": 0.95,
        "context_awareness": 0.9,
        "adaptability": 0.5,
        "engagement": 0.2,
        "personability": 0.2
    },
    "creative_thinker": {
        "empathy": 0.7,
        "technical_knowledge": 0.6,
        "creativity": 0.99,
        "conciseness": 0.4,
        "context_awareness": 0.7,
        "adaptability": 0.9,
        "engagement": 0.8,
        "personability": 0.7
    },
    "concise_analyst": {
        "empathy": 0.3,
        "technical_knowledge": 0.85,
        "creativity": 0.4,
        "conciseness": 0.99,
        "context_awareness": 0.8,
        "adaptability": 0.6,
        "engagement": 0.3,
        "personability": 0.3
    },
    "balanced": {
        "empathy": 0.5,
        "technical_knowledge": 0.7,
        "creativity": 0.5,
        "conciseness": 0.6,
        "context_awareness": 0.7,
        "adaptability": 0.7,
        "engagement": 0.5,
        "personability": 0.5
    },
    "high_context": {
        "empathy": 0.6,
        "technical_knowledge": 0.8,
        "creativity": 0.5,
        "conciseness": 0.7,
        "context_awareness": 0.99,
        "adaptability": 0.8,
        "engagement": 0.5,
        "personability": 0.5
    }
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("matrix_evaluation.log")
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class MatrixConfig:
    """Configuration for a single matrix evaluation run."""
    model: str
    personality: str  # "baseline", personality name, or "cot"
    benchmark: str
    traits: Optional[Dict[str, float]] = None
    cot_mode: bool = False


@dataclass
class MatrixResult:
    """Results for a single matrix configuration."""
    config: MatrixConfig
    score: Optional[float] = None
     # lm-eval returns dict with task results
    raw_results: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    error: Optional[str] = None
    timestamp: str = ""


class MatrixEvaluator:
    """Orchestrates matrix evaluation across models, personalities, and benchmarks."""

    def __init__(
        self,
        output_dir: str = "benchmark_results/matrix_evaluation",
        max_concurrency: int = 2,
        use_direct_ollama: bool = True  # Use direct Ollama API by default
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrency = max_concurrency
        self.use_direct_ollama = use_direct_ollama
        
        if use_direct_ollama:
            self.benchmark_runner = DirectOllamaBenchmarkRunner(
                output_dir=str(self.output_dir / "direct_ollama")
            )
        else:
            self.benchmark_runner = BenchmarkRunner(
                output_dir=str(self.output_dir / "lm_eval")
            )
        
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.results_file = self.output_dir / "results.json"
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrency)

    def generate_matrix_config(
        self,
        models: Optional[List[str]] = None,
        personalities: Optional[List[str]] = None,
        benchmarks: Optional[List[str]] = None,
        min_size: float = 0.5,
        max_size: float = 3.0
    ) -> List[MatrixConfig]:
        """Generate all combinations of model × personality × benchmark.
        
        Args:
            models: Optional list of models (if None, auto-detect)
            personalities: Optional list of personality names (if None, use all)
            benchmarks: Optional list of benchmarks (if None, use standard set)
            min_size: Minimum model parameter count in billions
            max_size: Maximum model parameter count in billions
        
        Returns:
            List of MatrixConfig objects representing all combinations
        """
        # Auto-detect models if not provided
        if models is None:
            logger.info(f"Auto-detecting models in range [{min_size}B, {max_size}B]...")
            try:
                models = detect_ollama_models_in_range(min_size=min_size, max_size=max_size)
            except Exception as e:
                logger.error(f"Failed to detect models: {e}")
                raise
            if not models:
                logger.warning("No models detected. Please specify models manually with --models")
                return []
            logger.info(f"Detected {len(models)} models: {models}")
        
        # Validate model availability (skip in test environments)
        import os
        skip_validation = os.getenv("SKIP_MODEL_VALIDATION", "false").lower() == "true"
        
        if not skip_validation:
            logger.info("Validating model availability...")
            from scripts.detect_model_sizes import validate_model_available
            validated_models = []
            for model in models:
                try:
                    if validate_model_available(model):
                        validated_models.append(model)
                        logger.debug(f"✓ Model available: {model}")
                    else:
                        logger.warning(f"✗ Model not available: {model}")
                except Exception as e:
                    logger.warning(f"✗ Error checking model {model}: {e}")
            
            if not validated_models:
                logger.error("No validated models available. Please check Ollama is running and models are pulled.")
                raise ValueError("No validated models available")
            
            if len(validated_models) < len(models):
                logger.warning(f"Only {len(validated_models)}/{len(models)} models are available. Continuing with available models.")
            
            models = validated_models
        else:
            logger.debug("Skipping model validation (SKIP_MODEL_VALIDATION=true)")

        # Use all personalities if not specified
        if personalities is None:
            # Generate all personality combinations:
            # 1. baseline
            # 2. All personality archetypes (without COT)
            # 3. All personality archetypes (with COT)
            # 4. cot (baseline + COT)
            base_personalities = [k for k in PERSONALITY_ARCHETYPES.keys() if k != "baseline"]
            personalities = []
            
            # Add baseline
            personalities.append("baseline")
            
            # Add all personality archetypes without COT
            for p in base_personalities:
                personalities.append(p)
            
            # Add all personality archetypes with COT
            for p in base_personalities:
                personalities.append(f"{p}+cot")
            
            # Add standalone COT
            personalities.append("cot")
        
        # Use standard benchmarks if not specified
        if benchmarks is None:
            benchmarks = ["mmlu", "hellaswag", "gsm8k", "arc_easy", "truthfulqa_mc2"]

        # Generate all combinations
        configs = []
        for model in models:
            for personality in personalities:
                for benchmark in benchmarks:
                    # Determine traits and COT mode
                    traits = None
                    cot_mode = False
                    
                    if personality == "cot":
                        # Standalone COT (baseline + COT)
                        cot_mode = True
                        traits = None
                    elif personality == "baseline":
                        # Pure baseline
                        traits = None
                        cot_mode = False
                    elif personality == "baseline+cot":
                        # Redundant - same as "cot", but handle gracefully
                        logger.debug(f"Converting 'baseline+cot' to 'cot' for {model}::{benchmark}")
                        cot_mode = True
                        traits = None
                    elif personality.endswith("+cot"):
                        # Personality + COT combination
                        if personality.count("+cot") > 1:
                            logger.warning(f"Invalid personality format (multiple +cot): {personality}, skipping")
                            continue
                        
                        base_personality = personality[:-4]  # Remove "+cot" suffix
                        
                        if base_personality == "baseline":
                            # baseline+cot is redundant, treat as cot
                            cot_mode = True
                            traits = None
                        elif base_personality in PERSONALITY_ARCHETYPES:
                            traits = PERSONALITY_ARCHETYPES[base_personality]
                            cot_mode = True
                        else:
                            logger.warning(f"Unknown base personality for {personality}: {base_personality}, skipping")
                            continue
                    elif personality in PERSONALITY_ARCHETYPES:
                        # Personality without COT
                        traits = PERSONALITY_ARCHETYPES[personality]
                        cot_mode = False
                    else:
                        logger.warning(f"Unknown personality: {personality}, skipping")
                        continue
                    
                    config = MatrixConfig(
                        model=model,
                        personality=personality,
                        benchmark=benchmark,
                        traits=traits,
                        cot_mode=cot_mode
                    )
                    configs.append(config)

        logger.info(f"Generated {len(configs)} matrix configurations")
        logger.info(f"  Models: {len(models)}")
        logger.info(f"  Personalities: {len(personalities)}")
        logger.info(f"  Benchmarks: {len(benchmarks)}")
        logger.info(f"  Total combinations: {len(models)} × {len(personalities)} × {len(benchmarks)} = {len(configs)}")

        return configs

    def load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint if it exists."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return {"completed": [], "results": []}

    def save_checkpoint(self, completed: List[str], results: List[Dict[str, Any]]):
        """Save checkpoint with completed configurations and results."""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "completed": completed,
            "results": results
        }
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            logger.debug(f"Checkpoint saved: {len(completed)} completed")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def get_config_key(self, config: MatrixConfig) -> str:
        """Generate unique key for a configuration."""
        return f"{config.model}::{config.personality}::{config.benchmark}"

    async def run_matrix_evaluation(
        self,
        configs: List[MatrixConfig],
        limit: Optional[int] = None,
        resume: bool = False
    ) -> List[MatrixResult]:
        """Run matrix evaluation for all configurations.
        
        Args:
            configs: List of matrix configurations to evaluate
            limit: Limit number of samples per benchmark
            resume: If True, skip already-completed configurations
        
        Returns:
            List of MatrixResult objects
        """
        # Load checkpoint if resuming
        completed_keys = set()
        existing_results = []
        if resume:
            checkpoint = self.load_checkpoint()
            completed_keys = set(checkpoint.get("completed", []))
            existing_results = checkpoint.get("results", [])
            logger.info(f"Resuming: {len(completed_keys)} configurations already completed")

        # Filter out completed configurations
        remaining_configs = [
            config for config in configs
            if self.get_config_key(config) not in completed_keys
        ]
        
        if not remaining_configs:
            logger.info("All configurations already completed!")
            return []

        total_remaining = len(remaining_configs)
        total_all_configs = len(configs)  # Total including already completed
        logger.info(f"Running {total_remaining} configurations...")
        logger.info(f"Progress: {len(existing_results)}/{total_all_configs} ({len(existing_results)/total_all_configs*100 if total_all_configs > 0 else 0:.1f}%)")

        # Execute in parallel batches
        results = []
        completed_keys_list = list(completed_keys)
        
        # Process in batches with concurrency limit
        for i in range(0, len(remaining_configs), self.max_concurrency):
            batch = remaining_configs[i:i + self.max_concurrency]
            batch_num = i // self.max_concurrency + 1
            total_batches = (len(remaining_configs) + self.max_concurrency - 1) // self.max_concurrency
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} configurations)")
            
            batch_results = await asyncio.gather(
                *[self._run_single_config(config, limit) for config in batch],
                return_exceptions=True
            )
            
            for config, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error in {self.get_config_key(config)}: {result}")
                    result = MatrixResult(
                        config=config,
                        error=str(result),
                        timestamp=datetime.now().isoformat()
                    )
                
                results.append(result)
                completed_keys_list.append(self.get_config_key(config))
                
                # Log progress
                completed_count = len(results) + len(existing_results)
                progress_pct = (completed_count / total_all_configs * 100) if total_all_configs > 0 else 0
                status = "✓" if result.error is None else "✗"
                logger.info(f"Progress: {completed_count}/{total_all_configs} ({progress_pct:.1f}%) - {status} {self.get_config_key(config)}")
                
                # Save checkpoint after each batch
                try:
                    self.save_checkpoint(
                        completed_keys_list,
                        [asdict(r) for r in results + existing_results]
                    )
                except Exception as e:
                    logger.warning(f"Failed to save checkpoint: {e}")

        # Combine with existing results
        all_results = existing_results + [asdict(r) for r in results]
        
        # Save final results
        self._save_results(all_results)
        
        # Convert dict results back to MatrixResult objects for return
        return [MatrixResult(**r) if isinstance(r, dict) else r for r in all_results]

    async def _run_single_config(
        self,
        config: MatrixConfig,
        limit: Optional[int] = None
    ) -> MatrixResult:
        """Run evaluation for a single matrix configuration."""
        start_time = time.time()
        config_key = self.get_config_key(config)
        
        logger.info(f"Running: {config_key}")
        
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            if self.use_direct_ollama:
                # Direct Ollama runner supports traits and COT mode
                result = await loop.run_in_executor(
                    None,
                    self.benchmark_runner.run_benchmark_task_sync,
                    config.model,
                    [config.benchmark],
                    limit,
                    0,  # num_fewshot
                    config.traits,  # Pass personality traits
                    config.cot_mode  # Pass COT mode
                )
            else:
                # Standard lm_eval runner
                result = await loop.run_in_executor(
                    None,
                    self.benchmark_runner.run_benchmark_task,
                    config.model,
                    [config.benchmark],
                    limit,
                    0  # num_fewshot
                )
            
            execution_time = time.time() - start_time
            
            if result:
                # Extract score from lm-eval results
                # lm-eval returns results in format: {"results": {"task": {"acc,none": score}}}
                task_results = result.get("results", {})
                if not task_results:
                    logger.warning(f"No results in response for {config_key}. Raw result: {result}")
                    return MatrixResult(
                        config=config,
                        error="No results in response",
                        raw_results=result,
                        execution_time=execution_time,
                        timestamp=datetime.now().isoformat()
                    )
                
                benchmark_results = task_results.get(config.benchmark, {})
                if not benchmark_results:
                    logger.warning(f"No results for benchmark {config.benchmark} in {config_key}. Available tasks: {list(task_results.keys())}")
                    return MatrixResult(
                        config=config,
                        error=f"No results for benchmark {config.benchmark}",
                        raw_results=result,
                        execution_time=execution_time,
                        timestamp=datetime.now().isoformat()
                    )
                
                # Try different possible keys for accuracy
                # MMLU uses "acc,none" format
                score = (
                    benchmark_results.get("acc,none") or
                    benchmark_results.get("acc") or
                    benchmark_results.get("acc_norm") or
                    benchmark_results.get("exact_match") or
                    None
                )
                
                if score is None:
                    logger.warning(f"Could not extract score from {config_key}. Available keys: {list(benchmark_results.keys())}")
            else:
                logger.warning(f"Benchmark returned None for {config_key}")
                score = None
            
            return MatrixResult(
                config=config,
                score=score,
                raw_results=result,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Error running {config_key}: {e}")
            return MatrixResult(
                config=config,
                error=str(e),
                execution_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )

    def _save_results(self, results: List[Any]):
        """Save final results to JSON file."""
        try:
            with open(self.results_file, 'w') as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "total_configurations": len(results),
                        "results": [asdict(r) if hasattr(r, '__dict__') else r for r in results]
                    },
                    f,
                    indent=2
                )
            logger.info(f"Results saved to {self.results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run multi-model personality matrix evaluation"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "comprehensive"],
        default="standard",
        help="Benchmark mode (default: standard)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to test (default: auto-detect 0.5b-3b)"
    )
    parser.add_argument(
        "--personalities",
        nargs="+",
        help="Specific personalities to test (default: all)"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        help="Specific benchmarks to test (default: standard set)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of samples per benchmark"
    )
    parser.add_argument(
        "--min-size",
        type=float,
        default=0.5,
        help="Minimum model parameter count in billions (default: 0.5)"
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=3.0,
        help="Maximum model parameter count in billions (default: 3.0)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Maximum parallel executions (default: 2)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results/matrix_evaluation",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Get benchmarks for mode
    benchmarks = BENCHMARK_TASKS.get(args.mode, BENCHMARK_TASKS["standard"])

    # Create evaluator
    evaluator = MatrixEvaluator(
        output_dir=args.output_dir,
        max_concurrency=args.concurrency
    )

    # Generate matrix
    configs = evaluator.generate_matrix_config(
        models=args.models,
        personalities=args.personalities,
        benchmarks=args.benchmarks or benchmarks,
        min_size=args.min_size,
        max_size=args.max_size
    )

    if not configs:
        logger.error("No configurations to evaluate. Exiting.")
        sys.exit(1)

    # Run evaluation
    logger.info("=" * 70)
    logger.info("MULTI-MODEL PERSONALITY MATRIX EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Total configurations: {len(configs)}")
    logger.info(f"Concurrency: {args.concurrency}")
    logger.info(f"Limit: {args.limit if args.limit else 'None'}")
    logger.info("=" * 70)

    results = asyncio.run(
        evaluator.run_matrix_evaluation(
            configs=configs,
            limit=args.limit,
            resume=args.resume
        )
    )

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total configurations: {len(results)}")
    successful = sum(1 for r in results if r.error is None)
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {len(results) - successful}")
    logger.info(f"Results saved to: {evaluator.results_file}")


if __name__ == "__main__":
    main()

