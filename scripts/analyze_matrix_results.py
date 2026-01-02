#!/usr/bin/env python3
"""
Matrix Results Analyzer

Analyzes results from multi-model personality matrix evaluation.
Calculates statistics, generates comparison tables, and identifies best configurations.

Usage:
    python scripts/analyze_matrix_results.py --input benchmark_results/matrix_evaluation/results.json
    python scripts/analyze_matrix_results.py --input results.json --output analysis.md
"""

import argparse
import json
import logging
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ConfigurationStats:
    """Statistics for a model-personality configuration."""
    model: str
    personality: str
    scores: List[float]
    mean_score: float
    std_dev: float
    min_score: float
    max_score: float
    vs_baseline: Optional[float] = None
    benchmark_scores: Dict[str, float] = None


@dataclass
class BestConfiguration:
    """Best performing configuration."""
    model: str
    personality: str
    average_score: float
    improvement_vs_baseline: float
    benchmark_breakdown: Dict[str, float]


class MatrixResultsAnalyzer:
    """Analyzes matrix evaluation results and generates reports."""

    def __init__(self, results_file: Path):
        self.results_file = Path(results_file)
        self.results_data = None
        self.results = []

    def load_results(self) -> bool:
        """Load results from JSON file."""
        if not self.results_file.exists():
            logger.error(f"Results file not found: {self.results_file}")
            return False

        try:
            with open(self.results_file, 'r') as f:
                self.results_data = json.load(f)
            
            # Extract results list
            if isinstance(self.results_data, dict) and "results" in self.results_data:
                self.results = self.results_data["results"]
            elif isinstance(self.results_data, list):
                self.results = self.results_data
            else:
                logger.error("Invalid results format")
                return False

            logger.info(f"Loaded {len(self.results)} result entries")
            return True
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return False

    def parse_results(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Parse results into nested structure: model -> personality -> benchmark -> score."""
        parsed = defaultdict(lambda: defaultdict(dict))

        for result in self.results:
            # Handle both dict and object-like results
            if isinstance(result, dict):
                config = result.get("config", {})
                score = result.get("score")
            else:
                # If it's a MatrixResult-like object
                config = getattr(result, "config", {})
                if isinstance(config, dict):
                    pass
                else:
                    config = {
                        "model": getattr(config, "model", ""),
                        "personality": getattr(config, "personality", ""),
                        "benchmark": getattr(config, "benchmark", "")
                    }
                score = getattr(result, "score", None) if hasattr(result, "score") else result.get("score")

            if score is None:
                continue

            model = config.get("model", "")
            personality = config.get("personality", "")
            benchmark = config.get("benchmark", "")

            if model and personality and benchmark:
                parsed[model][personality][benchmark] = score

        return parsed

    def calculate_statistics(
        self,
        parsed_results: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Dict[str, ConfigurationStats]]:
        """Calculate statistics per model-personality configuration."""
        stats = defaultdict(dict)

        # First pass: collect all scores
        for model, personalities in parsed_results.items():
            for personality, benchmarks in personalities.items():
                scores = list(benchmarks.values())
                if not scores:
                    continue

                mean_score = statistics.mean(scores)
                std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
                min_score = min(scores)
                max_score = max(scores)

                stats[model][personality] = ConfigurationStats(
                    model=model,
                    personality=personality,
                    scores=scores,
                    mean_score=mean_score,
                    std_dev=std_dev,
                    min_score=min_score,
                    max_score=max_score,
                    benchmark_scores=benchmarks.copy()
                )

        # Second pass: calculate vs baseline
        for model, personalities in stats.items():
            baseline_mean = None
            if "baseline" in personalities:
                baseline_mean = personalities["baseline"].mean_score

            for personality, stat in personalities.items():
                if personality != "baseline" and baseline_mean is not None:
                    stat.vs_baseline = stat.mean_score - baseline_mean

        return stats

    def generate_comparison_table(
        self,
        stats: Dict[str, Dict[str, ConfigurationStats]],
        benchmarks: List[str]
    ) -> str:
        """Generate Markdown comparison table."""
        md = "\n## Matrix Evaluation Results\n\n"
        md += "### Performance by Model × Personality × Benchmark\n\n"

        # Table header
        header = "| Model | Personality |"
        for bench in benchmarks:
            header += f" {bench} |"
        header += " Average | vs Baseline |\n"
        md += header

        # Separator
        separator = "|" + "|".join(["---"] * (len(benchmarks) + 4)) + "|\n"
        md += separator

        # Table rows
        for model in sorted(stats.keys()):
            for personality in sorted(stats[model].keys()):
                stat = stats[model][personality]
                row = f"| {model} | {personality} |"

                # Benchmark scores
                for bench in benchmarks:
                    score = stat.benchmark_scores.get(bench, None)
                    if score is not None:
                        row += f" {score:.2f} |"
                    else:
                        row += " - |"

                # Average
                row += f" {stat.mean_score:.2f} |"

                # vs Baseline
                if stat.vs_baseline is not None:
                    delta_str = f"+{stat.vs_baseline:.2f}" if stat.vs_baseline >= 0 else f"{stat.vs_baseline:.2f}"
                    row += f" {delta_str} |"
                else:
                    row += " - |"

                md += row + "\n"

        return md

    def calculate_cot_improvements(
        self,
        stats: Dict[str, Dict[str, ConfigurationStats]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate improvement when COT is added to each personality.
        
        Returns:
            Dict[model][personality] -> improvement percentage
        """
        improvements = defaultdict(dict)
        
        for model, personalities in stats.items():
            for personality, stat in personalities.items():
                if personality.endswith("+cot"):
                    base_personality = personality[:-4]  # Remove "+cot" suffix
                    if base_personality in personalities:
                        base_stat = personalities[base_personality]
                        improvement = stat.mean_score - base_stat.mean_score
                        improvement_pct = (improvement / base_stat.mean_score * 100) if base_stat.mean_score > 0 else 0
                        improvements[model][base_personality] = {
                            "absolute": improvement,
                            "percentage": improvement_pct,
                            "base_score": base_stat.mean_score,
                            "cot_score": stat.mean_score
                        }
        
        return improvements

    def find_best_configurations(
        self,
        stats: Dict[str, Dict[str, ConfigurationStats]],
        top_n: int = 10,
        rank_by: str = "average"
    ) -> List[BestConfiguration]:
        """Find best performing configurations.
        
        Args:
            stats: Statistics dictionary
            top_n: Number of top configurations to return
            rank_by: Ranking method - "average" (average score) or "improvement" (vs baseline)
        
        Returns:
            List of BestConfiguration objects, sorted by specified metric
        """
        candidates = []

        for model, personalities in stats.items():
            for personality, stat in personalities.items():
                if personality == "baseline":
                    continue  # Skip baseline in rankings

                candidates.append(BestConfiguration(
                    model=model,
                    personality=personality,
                    average_score=stat.mean_score,
                    improvement_vs_baseline=stat.vs_baseline or 0.0,
                    benchmark_breakdown=stat.benchmark_scores.copy()
                ))

        # Sort by specified metric
        if rank_by == "improvement":
            candidates.sort(key=lambda x: x.improvement_vs_baseline, reverse=True)
        else:  # default to average
            candidates.sort(key=lambda x: x.average_score, reverse=True)

        return candidates[:top_n]

    def generate_summary_report(
        self,
        stats: Dict[str, Dict[str, ConfigurationStats]],
        best_configs: List[BestConfiguration],
        benchmarks: List[str]
    ) -> str:
        """Generate comprehensive summary report."""
        md = "# Multi-Model Personality Matrix Analysis\n\n"
        md += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Summary statistics
        md += "## Summary Statistics\n\n"
        md += f"- Total configurations evaluated: {sum(len(p) for p in stats.values())}\n"
        md += f"- Models tested: {len(stats)}\n"
        md += f"- Benchmarks: {', '.join(benchmarks)}\n\n"

        # Best configurations
        md += "## Top Performing Configurations\n\n"
        md += "### Ranked by Average Score\n\n"
        md += "| Rank | Model | Personality | Avg Score | vs Baseline |\n"
        md += "|------|-------|------------|-----------|-------------|\n"

        for i, config in enumerate(best_configs, 1):
            delta_str = f"+{config.improvement_vs_baseline:.2f}" if config.improvement_vs_baseline >= 0 else f"{config.improvement_vs_baseline:.2f}"
            md += f"| {i} | {config.model} | {config.personality} | {config.average_score:.2f} | {delta_str} |\n"

        # COT Improvement Analysis
        cot_improvements = self.calculate_cot_improvements(stats)
        if cot_improvements:
            md += "\n## COT Prompting Impact\n\n"
            md += "Improvement when adding COT to each personality:\n\n"
            md += "| Model | Personality | Without COT | With COT | Improvement | % Change |\n"
            md += "|-------|-------------|-------------|----------|-------------|----------|\n"
            
            for model in sorted(cot_improvements.keys()):
                for base_personality in sorted(cot_improvements[model].keys()):
                    imp_data = cot_improvements[model][base_personality]
                    improvement_str = f"+{imp_data['absolute']:.2f}" if imp_data['absolute'] >= 0 else f"{imp_data['absolute']:.2f}"
                    pct_str = f"+{imp_data['percentage']:.1f}%" if imp_data['percentage'] >= 0 else f"{imp_data['percentage']:.1f}%"
                    md += f"| {model} | {base_personality} | {imp_data['base_score']:.2f} | {imp_data['cot_score']:.2f} | {improvement_str} | {pct_str} |\n"
            md += "\n"

        md += "\n### Detailed Breakdown\n\n"
        for i, config in enumerate(best_configs[:5], 1):  # Top 5 detailed
            md += f"#### {i}. {config.model} + {config.personality}\n\n"
            md += f"- **Average Score**: {config.average_score:.2f}\n"
            md += f"- **Improvement vs Baseline**: {config.improvement_vs_baseline:+.2f}\n"
            md += "- **Benchmark Breakdown**:\n"
            for bench, score in config.benchmark_breakdown.items():
                md += f"  - {bench}: {score:.2f}\n"
            md += "\n"

        # Comparison table
        md += self.generate_comparison_table(stats, benchmarks)

        # Statistics by model
        md += "\n## Statistics by Model\n\n"
        for model in sorted(stats.keys()):
            md += f"### {model}\n\n"
            md += "| Personality | Mean | Std Dev | Min | Max | vs Baseline |\n"
            md += "|-------------|------|---------|-----|-----|-------------|\n"

            for personality in sorted(stats[model].keys()):
                stat = stats[model][personality]
                vs_baseline_str = f"{stat.vs_baseline:+.2f}" if stat.vs_baseline is not None else "-"
                md += f"| {personality} | {stat.mean_score:.2f} | {stat.std_dev:.2f} | {stat.min_score:.2f} | {stat.max_score:.2f} | {vs_baseline_str} |\n"
            md += "\n"

        return md

    def analyze(self) -> Tuple[Dict[str, Dict[str, ConfigurationStats]], List[BestConfiguration]]:
        """Perform full analysis."""
        if not self.load_results():
            return {}, []

        # Parse results
        parsed = self.parse_results()
        logger.info(f"Parsed results for {len(parsed)} models")

        # Calculate statistics
        stats = self.calculate_statistics(parsed)
        logger.info(f"Calculated statistics for {sum(len(p) for p in stats.values())} configurations")

        # Find best configurations
        best_configs = self.find_best_configurations(stats)
        logger.info(f"Identified {len(best_configs)} top configurations")

        return stats, best_configs

    def export_markdown(
        self,
        stats: Dict[str, Dict[str, ConfigurationStats]],
        best_configs: List[BestConfiguration],
        output_path: Optional[Path] = None
    ) -> str:
        """Export analysis to Markdown format."""
        # Collect all benchmarks
        benchmarks = set()
        for model_stats in stats.values():
            for config_stats in model_stats.values():
                benchmarks.update(config_stats.benchmark_scores.keys())
        benchmarks = sorted(list(benchmarks))

        # Generate report
        report = self.generate_summary_report(stats, best_configs, benchmarks)

        # Write to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report written to {output_path}")

        return report


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze multi-model personality matrix evaluation results"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to results JSON file"
    )
    parser.add_argument(
        "--output",
        help="Path to output Markdown file (default: print to stdout)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top configurations to show (default: 10)"
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = MatrixResultsAnalyzer(args.input)

    # Perform analysis
    stats, best_configs = analyzer.analyze()

    if not stats:
        logger.error("No statistics calculated. Exiting.")
        sys.exit(1)

    # Export results
    report = analyzer.export_markdown(stats, best_configs, args.output)

    if not args.output:
        print(report)

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()

