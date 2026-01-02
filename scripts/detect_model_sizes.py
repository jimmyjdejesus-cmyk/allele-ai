#!/usr/bin/env python3
"""
Model Size Detection Utility

Detects available Ollama models and filters them by parameter count (0.5b-3b range).
Supports manual override for models that can't be auto-detected.
"""

import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_parameter_count(model_name: str) -> Optional[float]:
    """Parse parameter count from model name.
    
    Args:
        model_name: Model name (e.g., "qwen2.5:0.5b", "gemma3:1b", "llama3.2:1b")
    
    Returns:
        Parameter count in billions (e.g., 0.5, 1.0, 2.0, 3.0) or None if not found
    """
    # Pattern to match parameter counts: 0.5b, 1b, 2b, 3b, etc.
    # Also handles formats like "1.1b", "2.5b", "3.8b"
    patterns = [
        r'(\d+\.?\d*)\s*b\b',  # Matches "0.5b", "1b", "2.5b", "3b"
        r'(\d+\.?\d*)\s*billion',  # Matches "1 billion", "2.5 billion"
    ]
    
    model_lower = model_name.lower()
    
    for pattern in patterns:
        match = re.search(pattern, model_lower)
        if match:
            try:
                count = float(match.group(1))
                return count
            except ValueError:
                continue
    
    # Try to infer from common model name patterns
    # Handle special cases like "phi3:mini" (3.8B), "tinyllama" (1.1B)
    if 'tiny' in model_lower or 'mini' in model_lower:
        # These are typically small models, but exact size varies
        # Return None to require manual specification
        return None
    
    return None


# Cache for model list to avoid repeated subprocess calls
_model_list_cache: Optional[List[str]] = None


def detect_ollama_models(use_cache: bool = True) -> List[str]:
    """Detect available Ollama models by querying Ollama API or CLI.
    
    Args:
        use_cache: If True, use cached result if available
    
    Returns:
        List of available model names
    """
    global _model_list_cache
    
    # Return cached result if available
    if use_cache and _model_list_cache is not None:
        return _model_list_cache
    
    models = []
    
    try:
        # Try using ollama list command
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            # Parse output - format is typically:
            # NAME                ID              SIZE    MODIFIED
            # qwen2.5:0.5b       abc123...       500MB   2025-01-01
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:  # Skip header
                for line in lines[1:]:
                    parts = line.split()
                    if parts:
                        model_name = parts[0]
                        models.append(model_name)
            logger.info(f"Detected {len(models)} models from ollama list")
        else:
            logger.warning(f"ollama list failed: {result.stderr}")
            
    except FileNotFoundError:
        logger.error("ollama command not found. Is Ollama installed?")
    except subprocess.TimeoutExpired:
        logger.error("ollama list command timed out")
    except Exception as e:
        logger.error(f"Error detecting models: {e}")
    
    # Cache the result
    if use_cache:
        _model_list_cache = models
    
    return models


def filter_models_by_size(models: List[str], min_size: float = 0.5, max_size: float = 3.0) -> List[str]:
    """Filter models by parameter count range.
    
    Args:
        models: List of model names
        min_size: Minimum parameter count in billions (default: 0.5)
        max_size: Maximum parameter count in billions (default: 3.0)
    
    Returns:
        Filtered list of model names within the specified range
    """
    filtered = []
    ambiguous = []
    
    for model in models:
        param_count = parse_parameter_count(model)
        
        if param_count is None:
            ambiguous.append(model)
            logger.debug(f"Could not determine size for model: {model}")
            continue
        
        if min_size <= param_count <= max_size:
            filtered.append(model)
            logger.debug(f"Model {model} ({param_count}B) within range")
        else:
            logger.debug(f"Model {model} ({param_count}B) outside range [{min_size}B, {max_size}B]")
    
    if ambiguous:
        logger.info(f"Found {len(ambiguous)} models with ambiguous sizes (excluded): {ambiguous[:5]}")
    
    return filtered


def validate_model_available(model_name: str) -> bool:
    """Validate that a model is available in Ollama.
    
    Uses cached model list if available to avoid repeated subprocess calls.
    
    Args:
        model_name: Model name to check
    
    Returns:
        True if model is available, False otherwise
    """
    # Use cached model list if available
    models = detect_ollama_models(use_cache=True)
    return model_name in models


def detect_ollama_models_in_range(
    min_size: float = 0.5,
    max_size: float = 3.0,
    manual_models: Optional[List[str]] = None
) -> List[str]:
    """Detect Ollama models in specified parameter range.
    
    Args:
        min_size: Minimum parameter count in billions (default: 0.5)
        max_size: Maximum parameter count in billions (default: 3.0)
        manual_models: Optional list of manually specified models to include
    
    Returns:
        List of validated model names in the specified range
    """
    # Get all available models
    all_models = detect_ollama_models()
    
    if not all_models:
        logger.warning("No models detected. Using manual list if provided.")
        if manual_models:
            return [m for m in manual_models if validate_model_available(m)]
        return []
    
    # Filter by size
    filtered = filter_models_by_size(all_models, min_size, max_size)
    
    # Add manual models if provided
    if manual_models:
        for model in manual_models:
            if model not in filtered:
                # Validate manual model is available
                if validate_model_available(model):
                    filtered.append(model)
                    logger.info(f"Added manual model: {model}")
                else:
                    logger.warning(f"Manual model not available: {model}")
    
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for model in filtered:
        if model not in seen:
            seen.add(model)
            result.append(model)
    
    logger.info(f"Found {len(result)} models in range [{min_size}B, {max_size}B]: {result}")
    return result


def main():
    """CLI entry point for model detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect Ollama models by parameter size")
    parser.add_argument(
        "--min-size",
        type=float,
        default=0.5,
        help="Minimum parameter count in billions (default: 0.5)"
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=3.0,
        help="Maximum parameter count in billions (default: 3.0)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Manually specify models to include"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON array"
    )
    
    args = parser.parse_args()
    
    models = detect_ollama_models_in_range(
        min_size=args.min_size,
        max_size=args.max_size,
        manual_models=args.models
    )
    
    if args.json:
        print(json.dumps(models))
    else:
        for model in models:
            print(model)
    
    return 0 if models else 1


if __name__ == "__main__":
    sys.exit(main())

