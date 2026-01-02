#!/usr/bin/env python3
"""
Analyze lm-eval benchmark results and generate reports.
Parses JSON output from lm-eval and creates comparison tables.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchmarkAnalyzer:
    """Analyzes and visualizes benchmark results."""

    def __init__(self, results_dir: str = "benchmark_results/lm_eval"):
        self.results_dir = Path(results_dir)

    def load_latest_results(self, pattern: str = "consolidated_*.json") -> Dict[str, Any]:
        """Load the most recent consolidated results file."""
        files = list(self.results_dir.glob(pattern))
        if not files:
            logger.warning(f"No results found matching {pattern}")
            return {}

        latest_file = max(files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading results from {latest_file}")

        with open(latest_file) as f:
            return json.load(f)

    def load_all_runs(self) -> Dict[str, Any]:
        """Load all individual run results."""
        all_results = {}

        # Scan subdirectories for results.json
        for result_file in self.results_dir.glob("*/results.json"):
            try:
                # Directory name format: model_name_timestamp
                dir_name = result_file.parent.name
                model_name = "_".join(dir_name.split("_")[:-2]) # simple heuristic

                with open(result_file) as f:
                    data = json.load(f)
                    # Use model name from file if available, else directory
                    model = data.get("config", {}).get("model_args", "").split(",")[0].replace("model=", "")
                    if not model:
                        model = model_name

                    all_results[model] = data
            except Exception as e:
                logger.error(f"Error loading {result_file}: {e}")

        return all_results

    def parse_metrics(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Parse raw results into a DataFrame."""
        rows = []

        for model, data in results.items():
            if "results" not in data:
                continue

            row = {"Model": model}
            total_acc = 0
            count = 0

            for task, metrics in data["results"].items():
                # Extract primary metric
                score = 0.0
                if "acc" in metrics:
                    score = metrics["acc"] * 100
                elif "acc_norm" in metrics:
                    score = metrics["acc_norm"] * 100
                elif "exact_match" in metrics:
                    score = metrics["exact_match"] * 100
                elif "acc,none" in metrics: # new lm-eval format
                    score = metrics["acc,none"] * 100

                row[task] = score
                total_acc += score
                count += 1

            if count > 0:
                row["Average"] = total_acc / count

            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            # Sort by Average if available, else Model
            sort_col = "Average" if "Average" in df.columns else "Model"
            df = df.sort_values(sort_col, ascending=False)

        return df

    def generate_markdown_report(self, df: pd.DataFrame, output_path: Optional[str] = None) -> str:
        """Generate a markdown report from the DataFrame."""
        if df.empty:
            return "No results available."

        # Format floats
        df_formatted = df.round(2)

        md = "# LM-Eval Benchmark Results\n\n"
        md += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"

        md += "### Performance Comparison\n\n"
        md += df_formatted.to_markdown(index=False)

        md += "\n\n### Metric Definitions\n"
        md += "- **MMLU**: 5-shot accuracy on massive multitask language understanding\n"
        md += "- **HellaSwag**: Zero-shot accuracy on commonsense reasoning\n"
        md += "- **GSM8K**: 5-shot exact match on grade school math\n"

        if output_path:
            with open(output_path, 'w') as f:
                f.write(md)
            logger.info(f"Report saved to {output_path}")

        return md

def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--output", default="benchmark_results/lm_eval/COMPARISON.md",
                        help="Output markdown file path")
    parser.add_argument("--source", choices=["latest", "all"], default="latest",
                        help="Source of results (latest consolidated or all folders)")

    args = parser.parse_args()

    analyzer = BenchmarkAnalyzer()

    if args.source == "latest":
        results = analyzer.load_latest_results()
    else:
        results = analyzer.load_all_runs()

    if not results:
        print("No results found.")
        return

    df = analyzer.parse_metrics(results)
    print("\nBenchmark Summary:")
    print(df.round(2).to_string(index=False))

    analyzer.generate_markdown_report(df, args.output)

if __name__ == "__main__":
    main()

