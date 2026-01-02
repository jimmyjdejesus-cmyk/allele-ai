#!/usr/bin/env python3
"""
Full Automation Pipeline for Mass LLM Benchmarking.
Orchestrates the entire process: running benchmarks, analyzing results, and updating docs.
"""

import argparse
import logging
import subprocess
import sys
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pipeline")

def run_command(cmd, description):
    """Run a shell command with logging."""
    logger.info(f"Starting: {description}")
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description} (Exit Code: {e.returncode})")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run full benchmark pipeline")
    parser.add_argument("--mode", choices=["quick", "standard", "comprehensive"], default="quick",
                        help="Benchmark depth")
    parser.add_argument("--models", nargs="+", help="Specific models to test (default: all recommended)")
    parser.add_argument("--auto-update-readme", action="store_true", help="Update README.md automatically")
    parser.add_argument("--limit", type=int, help="Limit samples per task")

    args = parser.parse_args()

    start_time = time.time()
    logger.info("🚀 Starting Benchmark Pipeline")

    # Step 1: Run Benchmarks
    cmd_runner = ["python3", "scripts/run_lm_eval_mass.py", "--mode", args.mode]
    if args.models:
        cmd_runner.extend(["--models"] + args.models)
    if args.limit:
        cmd_runner.extend(["--limit", str(args.limit)])

    if not run_command(cmd_runner, "Mass Benchmark Runner"):
        sys.exit(1)

    # Step 2: Analyze Results
    cmd_analysis = ["python3", "scripts/analyze_lm_eval_results.py"]
    if not run_command(cmd_analysis, "Result Analysis"):
        sys.exit(1)

    # Step 3: Update README (Optional)
    if args.auto_update_readme:
        cmd_update = ["python3", "scripts/update_readme_benchmarks.py"]
        run_command(cmd_update, "README Update")

    duration = time.time() - start_time
    logger.info(f"✅ Pipeline Completed in {duration/60:.1f} minutes")

if __name__ == "__main__":
    main()

