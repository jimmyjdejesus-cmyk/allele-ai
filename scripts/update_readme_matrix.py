#!/usr/bin/env python3
"""
Update README with Matrix Evaluation Results

Automatically updates README.md with matrix evaluation results.
Uses marker-based content insertion to preserve manual content.

Usage:
    python scripts/update_readme_matrix.py --input benchmark_results/matrix_evaluation/results.json
    python scripts/update_readme_matrix.py --input results.json --analyzer-output analysis.md
"""

import argparse
import json
import logging
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReadmeMatrixUpdater:
    """Updates README with matrix evaluation results."""

    def __init__(
        self,
        readme_path: str = "README.md",
        results_path: Optional[str] = None,
        analyzer_output: Optional[str] = None
    ):
        self.readme_path = Path(readme_path)
        self.results_path = Path(results_path) if results_path else None
        self.analyzer_output = Path(analyzer_output) if analyzer_output else None
        self.marker_start = "<!-- MATRIX_EVALUATION_START -->"
        self.marker_end = "<!-- MATRIX_EVALUATION_END -->"

    def backup_readme(self) -> bool:
        """Create backup of README before update."""
        if not self.readme_path.exists():
            logger.error(f"README not found: {self.readme_path}")
            return False

        backup_path = self.readme_path.with_suffix(
            f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        try:
            shutil.copy2(self.readme_path, backup_path)
            logger.info(f"Backup created: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False

    def generate_section_content(self) -> str:
        """Generate matrix evaluation section content from analysis results."""
        content = "\n## Multi-Model Personality Matrix Evaluation\n\n"
        content += "Comprehensive evaluation testing multiple small language models (0.5B-3B parameters) "
        content += "across different personality configurations and standardized benchmarks.\n\n"

        # If analyzer output is provided, use it
        if self.analyzer_output and self.analyzer_output.exists():
            logger.info(f"Using analyzer output from {self.analyzer_output}")
            try:
                analyzer_content = self.analyzer_output.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decode failed for {self.analyzer_output}, trying with errors='replace'")
                analyzer_content = self.analyzer_output.read_text(encoding='utf-8', errors='replace')
            
            # Extract summary statistics
            summary_match = re.search(r"## Summary Statistics\n\n(.*?)\n\n##", analyzer_content, re.DOTALL)
            if summary_match:
                content += "### Summary\n\n"
                content += summary_match.group(1).strip() + "\n\n"
            
            # Extract top performers
            top_configs_match = re.search(
                r"### Ranked by Average Score\n\n(.*?)\n\n### Detailed",
                analyzer_content,
                re.DOTALL
            )
            if top_configs_match:
                content += "### Top Performing Configurations\n\n"
                content += top_configs_match.group(1).strip() + "\n\n"
            
            # Extract comparison table
            comparison_match = re.search(
                r"## Matrix Evaluation Results\n\n(.*?)(?=\n## |$)",
                analyzer_content,
                re.DOTALL
            )
            if comparison_match:
                content += "### Full Results Matrix\n\n"
                content += comparison_match.group(1).strip() + "\n\n"
            
            # Add reference to full analysis
            try:
                rel_path = self.analyzer_output.relative_to(Path.cwd())
                content += f"*Full analysis: [`{self.analyzer_output.name}`]({rel_path})*\n\n"
            except ValueError:
                # Path is not relative to cwd (e.g., temp file in tests)
                content += f"*Full analysis: [`{self.analyzer_output.name}`]({self.analyzer_output})*\n\n"
        else:
            # Fallback: generate basic content
            content += "**Status**: Results pending. Run matrix evaluation to generate results.\n\n"
            content += "### Quick Start\n\n"
            content += "```bash\n"
            content += "# Run matrix evaluation\n"
            content += "python scripts/run_matrix_evaluation.py --mode standard --limit 50\n\n"
            content += "# Analyze results\n"
            content += "python scripts/analyze_matrix_results.py --input benchmark_results/matrix_evaluation/results.json --output analysis.md\n\n"
            content += "# Update README\n"
            content += "python scripts/update_readme_matrix.py --input benchmark_results/matrix_evaluation/results.json --analyzer-output analysis.md\n"
            content += "```\n\n"

        return content

    def update_readme(self) -> bool:
        """Update README with matrix results."""
        if not self.readme_path.exists():
            logger.error(f"README not found: {self.readme_path}")
            return False

        # Create backup
        if not self.backup_readme():
            logger.warning("Backup failed, but continuing with update")

        # Read current content
        try:
            content = self.readme_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to read README: {e}")
            return False

        # Generate new section content
        new_content = self.generate_section_content()

        # Prepare full section with markers
        full_section = f"{self.marker_start}\n{new_content}{self.marker_end}"

        # Check if markers exist - use string replacement instead of regex to avoid escape issues
        if self.marker_start in content and self.marker_end in content:
            # Find the start and end positions
            start_pos = content.find(self.marker_start)
            end_pos = content.find(self.marker_end, start_pos) + len(self.marker_end)
            
            # Replace the section
            content = content[:start_pos] + full_section + content[end_pos:]
            logger.info("Updated existing matrix evaluation section in README")
        else:
            # Find a good insertion point - after other benchmark sections
            # Look for PERSONALITY_RESULTS_END or LM_EVAL_RESULTS_END
            insert_after_patterns = [
                r"(<!-- PERSONALITY_RESULTS_END -->)",
                r"(<!-- LM_EVAL_RESULTS_END -->)",
                r"(## Benchmarks.*?\n)",
                r"(## Contributing)"
            ]
            
            inserted = False
            for pattern in insert_after_patterns:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    insert_point = match.end()
                    content = content[:insert_point] + "\n" + full_section + "\n" + content[insert_point:]
                    logger.info(f"Inserted matrix evaluation section after {pattern}")
                    inserted = True
                    break
            
            if not inserted:
                # Last resort: append to end
                content += "\n\n" + full_section
                logger.warning("Appended matrix evaluation section to end (no insertion point found)")

        # Validate Markdown (basic check)
        if not self._validate_markdown(content):
            logger.warning("Markdown validation warnings detected")

        # Write updated content
        try:
            self.readme_path.write_text(content, encoding='utf-8')
            logger.info(f"Successfully updated {self.readme_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to write README: {e}")
            return False

    def _validate_markdown(self, content: str) -> bool:
        """Basic Markdown validation."""
        # Check for balanced markers
        start_count = content.count(self.marker_start)
        end_count = content.count(self.marker_end)
        
        if start_count != end_count:
            logger.error(f"Unbalanced markers: {start_count} start, {end_count} end")
            return False

        # Check for table formatting (basic)
        table_lines = [line for line in content.split('\n') if '|' in line and line.strip().startswith('|')]
        if table_lines:
            # Check header separator
            has_separator = any(re.match(r'^\|[\s\-:]+\|', line) for line in table_lines)
            if not has_separator and len(table_lines) > 1:
                logger.warning("Table may be missing header separator")

        return True


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Update README with matrix evaluation results"
    )
    parser.add_argument(
        "--input",
        help="Path to matrix results JSON file (optional if using --analyzer-output)"
    )
    parser.add_argument(
        "--analyzer-output",
        help="Path to analyzer output Markdown file (from analyze_matrix_results.py)"
    )
    parser.add_argument(
        "--readme",
        default="README.md",
        help="Path to README file"
    )

    args = parser.parse_args()

    if not args.input and not args.analyzer_output:
        logger.error("Either --input or --analyzer-output must be provided")
        sys.exit(1)

    # Create updater
    updater = ReadmeMatrixUpdater(
        readme_path=args.readme,
        results_path=args.input,
        analyzer_output=args.analyzer_output
    )

    # Perform update
    if updater.update_readme():
        logger.info("README update complete!")
        sys.exit(0)
    else:
        logger.error("README update failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

