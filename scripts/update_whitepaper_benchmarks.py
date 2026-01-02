#!/usr/bin/env python3
"""
Update Whitepaper with Matrix Evaluation Results

Automatically updates the whitepaper Section 4.2.5 with matrix evaluation results.
Uses marker-based content insertion to preserve manual content.

Usage:
    python scripts/update_whitepaper_benchmarks.py --input benchmark_results/matrix_evaluation/results.json
    python scripts/update_whitepaper_benchmarks.py --input results.json --analyzer-output analysis.md
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


class WhitepaperUpdater:
    """Updates whitepaper with matrix evaluation results."""

    def __init__(
        self,
        whitepaper_path: str = "docs/whitepaper/phylogenic_whitepaper.md",
        results_path: Optional[str] = None,
        analyzer_output: Optional[str] = None
    ):
        self.whitepaper_path = Path(whitepaper_path)
        self.results_path = Path(results_path) if results_path else None
        self.analyzer_output = Path(analyzer_output) if analyzer_output else None
        self.marker_start = "<!-- MATRIX_RESULTS_START -->"
        self.marker_end = "<!-- MATRIX_RESULTS_END -->"

    def backup_whitepaper(self) -> bool:
        """Create backup of whitepaper before update."""
        if not self.whitepaper_path.exists():
            logger.error(f"Whitepaper not found: {self.whitepaper_path}")
            return False

        backup_path = self.whitepaper_path.with_suffix(
            f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        try:
            shutil.copy2(self.whitepaper_path, backup_path)
            logger.info(f"Backup created: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False

    def generate_section_content(self) -> str:
        """Generate Section 4.2.5 content from analysis results."""
        content = "\n### 4.2.5 Multi-Model Personality Evaluation\n\n"
        content += "This section presents results from comprehensive matrix evaluation "
        content += "testing multiple small language models (0.5B-3B parameters) across "
        content += "different personality configurations and standardized benchmarks.\n\n"

        # If analyzer output is provided, use it
        if self.analyzer_output and self.analyzer_output.exists():
            logger.info(f"Using analyzer output from {self.analyzer_output}")
            try:
                analyzer_content = self.analyzer_output.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                # Fallback: try with error handling
                logger.warning(f"UTF-8 decode failed, trying with errors='replace'")
                analyzer_content = self.analyzer_output.read_text(encoding='utf-8', errors='replace')
            
            # Sanitize content: replace backslashes with forward slashes to avoid regex escape issues
            # This is safe because we're only extracting text content, not file paths
            analyzer_content_safe = analyzer_content.replace('\\', '/')
            
            # Extract relevant sections from analyzer output
            # Get summary and top configurations
            summary_match = re.search(r"## Summary Statistics\n\n(.*?)\n\n##", analyzer_content_safe, re.DOTALL)
            top_configs_match = re.search(r"## Top Performing Configurations\n\n(.*?)\n\n### Detailed", analyzer_content_safe, re.DOTALL)
            # Use raw string to avoid escape issues
            comparison_match = re.search(r"## Matrix Evaluation Results\n\n(.*?)(?=\n## |$)", analyzer_content_safe, re.DOTALL)
            
            if summary_match:
                # Replace backslashes in extracted content to avoid regex issues
                summary_text = summary_match.group(1).strip().replace('\\', '/')
                content += summary_text + "\n\n"
            
            if top_configs_match:
                content += "#### Top Performers\n\n"
                # Replace backslashes in extracted content to avoid regex issues
                top_configs_text = top_configs_match.group(1).strip().replace('\\', '/')
                content += top_configs_text + "\n\n"
            
            if comparison_match:
                content += "#### Full Results Matrix\n\n"
                # Replace backslashes in extracted content to avoid regex issues
                comparison_text = comparison_match.group(1).strip().replace('\\', '/')
                content += comparison_text + "\n\n"
            
            # Add reference to full analysis
            try:
                rel_path = self.analyzer_output.relative_to(Path.cwd())
                # Use forward slashes to avoid regex escape issues
                rel_path_str = str(rel_path).replace('\\', '/')
                content += f"*Full analysis available in: `{rel_path_str}`*\n\n"
            except ValueError:
                # Path is not relative to cwd (e.g., temp file in tests)
                # Use forward slashes to avoid regex escape issues
                path_str = str(self.analyzer_output).replace('\\', '/')
                content += f"*Full analysis available in: `{path_str}`*\n\n"
        else:
            # Fallback: generate basic content
            content += "**Status**: Results pending. Run matrix evaluation to generate results.\n\n"
            content += "To generate results:\n"
            content += "```bash\n"
            content += "python scripts/run_matrix_evaluation.py --mode standard --limit 50\n"
            content += "python scripts/analyze_matrix_results.py --input benchmark_results/matrix_evaluation/results.json --output analysis.md\n"
            content += "python scripts/update_whitepaper_benchmarks.py --input benchmark_results/matrix_evaluation/results.json --analyzer-output analysis.md\n"
            content += "```\n\n"

        return content

    def update_whitepaper(self) -> bool:
        """Update whitepaper with matrix results."""
        if not self.whitepaper_path.exists():
            logger.error(f"Whitepaper not found: {self.whitepaper_path}")
            return False

        # Create backup
        if not self.backup_whitepaper():
            logger.warning("Backup failed, but continuing with update")

        # Read current content
        try:
            content = self.whitepaper_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to read whitepaper: {e}")
            return False

        # Generate new section content
        new_content = self.generate_section_content()

        # Prepare full section with markers
        # Replace backslashes in paths to avoid regex escape issues on Windows
        new_content_safe = new_content.replace('\\', '/')
        full_section = f"{self.marker_start}\n{new_content_safe}{self.marker_end}"

        # Check if markers exist - use string replacement instead of regex to avoid escape issues
        if self.marker_start in content and self.marker_end in content:
            # Find the start and end positions
            start_pos = content.find(self.marker_start)
            end_pos = content.find(self.marker_end, start_pos) + len(self.marker_end)
            
            # Replace the section
            content = content[:start_pos] + full_section + content[end_pos:]
            logger.info("Updated existing matrix results section")
        else:
            # Find Section 4.2 and insert after it
            section_42_pattern = r"(### 4\.2\s+Results.*?)(?=\n### 4\.3|\n## 5|$)"
            match = re.search(section_42_pattern, content, re.DOTALL)
            
            if match:
                # Insert after Section 4.2 content
                insert_point = match.end()
                content = content[:insert_point] + "\n" + full_section + "\n" + content[insert_point:]
                logger.info("Inserted matrix results section in Section 4.2")
            else:
                # Fallback: append before Section 5
                section_5_marker = "## 5."
                if section_5_marker in content:
                    # Use string replacement instead of regex
                    insert_pos = content.find(section_5_marker)
                    content = content[:insert_pos] + full_section + "\n\n" + content[insert_pos:]
                    logger.info("Inserted matrix results section before Section 5")
                else:
                    # Last resort: append to end
                    content += "\n\n" + full_section
                    logger.warning("Appended matrix results section to end (markers not found)")

        # Validate Markdown (basic check)
        if not self._validate_markdown(content):
            logger.warning("Markdown validation warnings detected")

        # Write updated content
        try:
            self.whitepaper_path.write_text(content, encoding='utf-8')
            logger.info(f"Successfully updated {self.whitepaper_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to write whitepaper: {e}")
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
        description="Update whitepaper with matrix evaluation results"
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
        "--whitepaper",
        default="docs/whitepaper/phylogenic_whitepaper.md",
        help="Path to whitepaper file"
    )

    args = parser.parse_args()

    if not args.input and not args.analyzer_output:
        logger.error("Either --input or --analyzer-output must be provided")
        sys.exit(1)

    # Create updater
    updater = WhitepaperUpdater(
        whitepaper_path=args.whitepaper,
        results_path=args.input,
        analyzer_output=args.analyzer_output
    )

    # Perform update
    if updater.update_whitepaper():
        logger.info("Whitepaper update complete!")
        sys.exit(0)
    else:
        logger.error("Whitepaper update failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

