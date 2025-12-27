"""
Shared utilities for benchmark evaluation and testing.

This module provides robust answer checking, system prompt building,
and other common functionality used across benchmark scripts.
"""

import re
from typing import Dict, List, Optional


def check_answer(response: str, expected: str) -> bool:
    """
    Robustly check if model response contains the expected answer.

    Handles multiple choice questions (A, B, C, D) and numerical answers.
    Fixes the issue where 'expected in response' could cause false positives
    for numerical answers (e.g., "16" incorrectly matching expected "6").

    Args:
        response: The model's response text
        expected: The expected answer

    Returns:
        True if response contains expected answer, False otherwise
    """
    response = response.strip()
    expected = expected.strip()

    # Handle multiple choice questions (A, B, C, D)
    if len(expected) == 1 and expected.upper() in "ABCD":
        # Look for patterns like "A", "A.", "(A)", "Answer: A", etc.
        # Use word boundaries to avoid partial matches
        pattern = rf'\b{re.escape(expected.upper())}\b'
        return bool(re.search(pattern, response.upper()))

    # For numerical answers, extract numbers and check for exact match
    try:
        float(expected)
        is_numeric_expected = True
    except ValueError:
        is_numeric_expected = False

    if is_numeric_expected:
        # Extract all numbers from response (including decimals and negatives)
        numbers_in_response = re.findall(r'-?\d+(?:\.\d+)?', response)
        return expected in numbers_in_response

    # For other text answers, do simple substring matching
    # Convert to lowercase for case-insensitive comparison
    return expected.lower() in response.lower()


def build_system_prompt(traits: Optional[Dict[str, float]]) -> str:
    """
    Build system prompt from genome traits.

    Args:
        traits: Dictionary of trait names to values (0.0 to 1.0)

    Returns:
        Formatted system prompt string
    """
    if not traits:
        return ""

    # Map trait values to behavioral descriptors
    trait_descriptions = []

    if traits.get("empathy", 0.5) > 0.7:
        trait_descriptions.append("Show deep understanding and emotional intelligence")
    if traits.get("technical_knowledge", 0.5) > 0.7:
        trait_descriptions.append("Provide technically accurate and detailed explanations")
    if traits.get("creativity", 0.5) > 0.7:
        trait_descriptions.append("Think creatively and offer novel perspectives")
    if traits.get("conciseness", 0.5) > 0.7:
        trait_descriptions.append("Be direct and concise - give short, precise answers")
    if traits.get("context_awareness", 0.5) > 0.7:
        trait_descriptions.append("Maintain strong awareness of context and implications")
    if traits.get("adaptability", 0.5) > 0.7:
        trait_descriptions.append("Adapt communication style to the task requirements")
    if traits.get("engagement", 0.5) > 0.7:
        trait_descriptions.append("Be engaging and thorough")
    if traits.get("personability", 0.5) > 0.7:
        trait_descriptions.append("Be friendly and approachable")

    if not trait_descriptions:
        return ""

    prompt = "You are an AI assistant. Your behavioral guidelines:\n"
    for desc in trait_descriptions:
        prompt += f"- {desc}\n"

    prompt += "\nFor multiple choice: answer with just the letter. For math: show work then final number."
    return prompt


def validate_trait_values(traits: Dict[str, float]) -> bool:
    """
    Validate that trait values are within valid range.

    Args:
        traits: Dictionary of trait names to values

    Returns:
        True if all values are valid (0.0 to 1.0), False otherwise
    """
    valid_traits = {
        "empathy", "technical_knowledge", "creativity", "conciseness",
        "context_awareness", "adaptability", "engagement", "personability"
    }

    for trait, value in traits.items():
        if trait not in valid_traits:
            raise ValueError(f"Unknown trait: {trait}")
        if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
            raise ValueError(f"Trait {trait} must be a number between 0.0 and 1.0, got {value}")

    return True


def format_benchmark_results(results: List[Dict], model_name: str) -> str:
    """
    Format benchmark results into a markdown table.

    Args:
        results: List of benchmark result dictionaries
        model_name: Name of the model tested

    Returns:
        Formatted markdown string
    """
    md = f"""
## Phylogenic Genome A/B Benchmark Results

**Model**: `{model_name}`
**Date**: 2025-12-19

### Performance Comparison

| Benchmark | Baseline | + Genome | Delta | Status |
|-----------|----------|----------|-------|--------|
"""

    for result in results:
        delta = result.get('delta', 0)
        if delta > 1:
            status = "[+] Improved"
        elif delta < -1:
            status = "[-] Degraded"
        else:
            status = "[=] Neutral"

        baseline = result.get('baseline_score', 0)
        phylogenic = result.get('phylogenic_score', 0)
        benchmark_name = result.get('benchmark_name', 'Unknown')

        md += f"| **{benchmark_name}** | {baseline:.1f}% | {phylogenic:.1f}% | {delta:+.1f}% | {status} |\n"

    return md
