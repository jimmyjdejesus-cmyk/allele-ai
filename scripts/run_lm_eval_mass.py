#!/usr/bin/env python3
"""
Mass LLM Benchmark Runner using lm-evaluation-harness
Optimized for M1 Mac with Ollama models.

Supports quick validation and full comprehensive benchmarking modes.
Handles model downloading, progress tracking, and results management.
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configuration
RECOMMENDED_MODELS = [
    "gemma2:2b",      # Google's 2B - excellent M1 performance (Tiny Gemma)
    "qwen2.5:0.5b",   # Alibaba's 0.5B - fastest
    "llama3.2:1b",    # Meta's 1B - balanced
    "phi3:mini",      # Microsoft's 3.8B - best quality/size
]

BENCHMARK_TASKS = {
    "quick": ["mmlu", "hellaswag", "gsm8k"],  # 3 core tasks
    "standard": ["mmlu", "hellaswag", "gsm8k", "arc_easy", "truthfulqa_mc2"],
    "comprehensive": [
        "mmlu", "hellaswag", "gsm8k", "arc_easy", "arc_challenge",
        "truthfulqa_mc2", "winogrande", "piqa", "boolq", "siqa"
    ]
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("benchmark_runner.log")
    ]
)
logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """Orchestrates mass LLM benchmarking using lm-eval."""

    def __init__(self, output_dir: str = "benchmark_results/lm_eval"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "http://localhost:11434"

    def check_ollama_running(self) -> bool:
        """Check if Ollama service is running."""
        try:
            result = subprocess.run(
                ["curl", "-s", f"{self.base_url}/api/version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False

    def ensure_model_available(self, model: str) -> bool:
        """Ensure model is downloaded. Handles Ollama pull or verifies HF access."""
        if model.startswith("hf:"):
            logger.info(f"Using Hugging Face model: {model[3:]} (will download if missing)")
            return True

        logger.info(f"Checking availability for model: {model}")

        # Check list first
        try:
            list_cmd = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if model in list_cmd.stdout:
                logger.info(f"Model {model} is already available")
                return True
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return False

        # Pull if not found
        logger.info(f"Pulling model {model}...")
        try:
            subprocess.run(["ollama", "pull", model], check=True)
            logger.info(f"Successfully pulled {model}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull model {model}: {e}")
            return False

    def run_benchmark_task(
        self,
        model: str,
        tasks: List[str],
        limit: Optional[int] = None,
        num_fewshot: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Run lm-eval for a specific model and task set."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = self.output_dir / f"{model.replace(':', '_').replace('/', '_')}_{timestamp}"

        # Determine backend and args
        if model.startswith("hf:"):
            # Hugging Face backend (e.g. for T5)
            real_model = model[3:]
            backend = "hf"
            # Use MPS for M1 acceleration. Only enable trust_remote_code when
            # explicitly requested via CLI flag due to remote code execution risk.
            trc = getattr(self, "trust_remote_code", False)
            if trc:
                logger.warning(
                    "Enabling trust_remote_code for Hugging Face models is unsafe. "
                    "Ensure you trust the model source before enabling this flag."
                )
            model_args = f"pretrained={real_model},device=mps,trust_remote_code={str(bool(trc))}"
        else:
            # Ollama backend (default)
            # Use local-completions with explicit tokenizer specification
            backend = "local-completions"
            # For Ollama, we need to specify a valid tokenizer model name
            # Use a generic tokenizer that works with most models
            # base_url should point to the server root, lm_eval will handle /v1
            # Note: local-completions may have issues with Ollama's /v1 endpoint
            # If this fails, consider using a proxy or different backend
            model_args = f"model={model},base_url={self.base_url}/v1,tokenizer=gpt2"

        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", backend,
            "--model_args", model_args,
            "--tasks", ",".join(tasks),
            "--output_path", str(run_output_dir),
            "--log_samples",
            "--num_fewshot", str(num_fewshot),
            "--batch_size", "1"  # M1 optimization
        ]

        if limit:
            cmd.extend(["--limit", str(limit)])

        logger.info(f"Starting benchmark for {model}")
        logger.info(f"Tasks: {tasks}")
        logger.info(f"Limit: {limit if limit else 'None'}")

        # Ensure output directory exists
        run_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            start_time = time.time()
            _result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = time.time() - start_time

            logger.info(f"Benchmark completed in {duration:.2f}s")

            # Load results
            results_file = run_output_dir / "results.json"
            if results_file.exists():
                with open(results_file, encoding='utf-8') as f:
                    return json.load(f)

            logger.warning(f"Benchmark completed but no results file found at {results_file}")
            return None

        except subprocess.CalledProcessError as e:
            logger.warning(f"Benchmark command failed for {model}, but checking for results file...")
            logger.debug(f"Command error: {e}")
            if e.stderr:
                logger.debug(f"STDERR: {e.stderr[:500]}")  # Limit stderr output
            
            # Try to extract results from output directory even if command failed
            # lm_eval may have written results before the error occurred
            results_file = run_output_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file, encoding='utf-8') as f:
                        results = json.load(f)
                        logger.info(f"✓ Successfully extracted results from {results_file} despite command error")
                        return results
                except json.JSONDecodeError as parse_error:
                    logger.error(f"Failed to parse results file {results_file}: {parse_error}")
                except Exception as read_error:
                    logger.error(f"Failed to read results file {results_file}: {read_error}")
            else:
                logger.debug(f"No results file found at {results_file}")
            
            logger.error(f"Benchmark failed for {model} and no recoverable results found")
            return None

    def run_mass_benchmarks(
        self,
        models: List[str],
        mode: str = "quick",
        custom_limit: Optional[int] = None,
        resume_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run mass benchmarks across multiple models."""

        if not self.check_ollama_running():
            logger.error("Ollama is not running! Please start it with 'ollama serve'")
            return {}

        tasks = BENCHMARK_TASKS.get(mode, BENCHMARK_TASKS["quick"])

        # Determine limit based on mode
        limit = custom_limit
        if limit is None:
            if mode == "quick":
                limit = 20
            elif mode == "standard":
                limit = None  # Full dataset

        all_results = {}

        # Load resume state if provided
        completed_models = []
        if resume_from and Path(resume_from).exists():
            with open(resume_from) as f:
                saved_state = json.load(f)
                all_results = saved_state.get("results", {})
                completed_models = list(all_results.keys())
                logger.info(f"Resuming run. Skipping: {completed_models}")

        for model in models:
            if model in completed_models:
                continue

            if not self.ensure_model_available(model):
                logger.warning(f"Skipping {model} due to unavailability")
                continue

            logger.info(f"Processing {model}...")
            results = self.run_benchmark_task(model, tasks, limit)

            if results:
                all_results[model] = results

                # Save checkpoint
                checkpoint_file = self.output_dir / "checkpoint_latest.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump({"mode": mode, "results": all_results}, f, indent=2)
            else:
                logger.error(f"Failed to get results for {model}")

        # Save consolidated results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        consolidated_file = self.output_dir / f"consolidated_{mode}_{timestamp}.json"

        with open(consolidated_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"All benchmarks complete. Results saved to {consolidated_file}")
        return all_results

def main():
    parser = argparse.ArgumentParser(description="Run mass LLM benchmarks")
    parser.add_argument("--mode", choices=["quick", "standard", "comprehensive"], default="quick",
                        help="Benchmark mode (default: quick)")
    parser.add_argument("--models", nargs="+", default=RECOMMENDED_MODELS,
                        help="List of models to benchmark")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading remote model code for Hugging Face models (unsafe). Disabled by default."
    )
    parser.add_argument("--limit", type=int, help="Override sample limit")
    parser.add_argument("--resume", help="Resume from checkpoint file")

    args = parser.parse_args()

    if "all" in args.models:
        models = RECOMMENDED_MODELS
    else:
        models = args.models

    runner = BenchmarkRunner()
    # propagate unsafe flag to runner
    runner.trust_remote_code = bool(args.trust_remote_code)
    runner.run_mass_benchmarks(
        models=models,
        mode=args.mode,
        custom_limit=args.limit,
        resume_from=args.resume
    )

if __name__ == "__main__":
    main()

