"""Tests for model size detection utility."""

import pytest
from unittest.mock import patch, MagicMock

from scripts.detect_model_sizes import (
    parse_parameter_count,
    filter_models_by_size,
    validate_model_available,
    detect_ollama_models_in_range
)


def test_parse_parameter_count_standard_formats():
    """Test parsing common model name formats."""
    assert parse_parameter_count("qwen2.5:0.5b") == 0.5
    assert parse_parameter_count("gemma3:1b") == 1.0
    assert parse_parameter_count("llama3.2:2b") == 2.0
    assert parse_parameter_count("phi3:3b") == 3.0
    assert parse_parameter_count("model:1.5b") == 1.5
    assert parse_parameter_count("model:2.5b") == 2.5


def test_parse_parameter_count_billion_format():
    """Test parsing 'billion' format."""
    assert parse_parameter_count("model:1 billion") == 1.0
    assert parse_parameter_count("model:2.5 billion") == 2.5


def test_parse_parameter_count_case_insensitive():
    """Test that parsing is case insensitive."""
    assert parse_parameter_count("MODEL:1B") == 1.0
    assert parse_parameter_count("Model:2B") == 2.0


def test_parse_parameter_count_ambiguous():
    """Test that ambiguous models return None."""
    assert parse_parameter_count("tinyllama") is None
    assert parse_parameter_count("phi3:mini") is None
    assert parse_parameter_count("unknown_model") is None


def test_filter_models_by_size():
    """Test filtering models by parameter count."""
    models = [
        "qwen2.5:0.5b",
        "gemma3:1b",
        "llama3.2:2b",
        "phi3:3b",
        "large:5b",
        "tinyllama"  # Ambiguous
    ]
    
    filtered = filter_models_by_size(models, min_size=0.5, max_size=3.0)
    assert "qwen2.5:0.5b" in filtered
    assert "gemma3:1b" in filtered
    assert "llama3.2:2b" in filtered
    assert "phi3:3b" in filtered
    assert "large:5b" not in filtered  # Too large
    assert "tinyllama" not in filtered  # Ambiguous


def test_filter_models_by_size_boundaries():
    """Test filtering respects boundaries."""
    models = ["model:0.5b", "model:3.0b", "model:0.4b", "model:3.1b"]
    filtered = filter_models_by_size(models, min_size=0.5, max_size=3.0)
    assert "model:0.5b" in filtered
    assert "model:3.0b" in filtered
    assert "model:0.4b" not in filtered
    assert "model:3.1b" not in filtered


@patch('scripts.detect_model_sizes.detect_ollama_models')
def test_validate_model_available_success(mock_detect):
    """Test successful model validation."""
    # Mock the cached model list
    mock_detect.return_value = ["qwen2.5:0.5b", "gemma3:1b"]
    assert validate_model_available("qwen2.5:0.5b") is True


@patch('scripts.detect_model_sizes.detect_ollama_models')
def test_validate_model_available_not_found(mock_detect):
    """Test model not found validation."""
    # Mock the cached model list
    mock_detect.return_value = ["other_model", "gemma3:1b"]
    assert validate_model_available("qwen2.5:0.5b") is False


@patch('scripts.detect_model_sizes.detect_ollama_models')
@patch('scripts.detect_model_sizes.validate_model_available')
def test_detect_ollama_models_in_range(mock_validate, mock_detect):
    """Test end-to-end model detection."""
    mock_detect.return_value = [
        "qwen2.5:0.5b",
        "gemma3:1b",
        "large:5b"
    ]
    mock_validate.return_value = True
    
    result = detect_ollama_models_in_range(
        min_size=0.5,
        max_size=3.0,
        manual_models=["manual:2b"]
    )
    
    assert "qwen2.5:0.5b" in result
    assert "gemma3:1b" in result
    assert "large:5b" not in result
    assert "manual:2b" in result

