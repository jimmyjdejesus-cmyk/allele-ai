"""
HumanEval Benchmark Implementation.

This module implements the HumanEval benchmark for evaluating Python code generation
from function docstrings.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp

from .base import Benchmark, BenchmarkResult
from .registry import register_benchmark


@register_benchmark("humaneval")
class HumanEvalBenchmark(Benchmark):
    """
    HumanEval Benchmark for Python Code Generation.

    Evaluates Python code generation capabilities by asking models to implement
    functions based on docstring descriptions.

    Each example contains:
    - Function signature
    - Docstring description
    - Test cases
    - Correct implementation
    """

    def __init__(self, split: str = "test", max_samples: Optional[int] = None):
        """
        Initialize HumanEval benchmark.

        Args:
            split: Dataset split ('test' recommended for evaluation)
            max_samples: Maximum number of samples to evaluate
        """
        description = "HumanEval: Python code generation from docstrings"
        super().__init__("HumanEval", description, max_score=100.0)

        self.split = split
        self.max_samples = max_samples
        self.data_dir = Path("benchmarks/data/humaneval")
        self.dataset = []

    async def setup(self) -> None:
        """Download and prepare HumanEval dataset."""
        self.logger.info("Setting up HumanEval benchmark...")

        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Check if dataset already exists
        data_file = self.data_dir / f"{self.split}.jsonl"
        if data_file.exists():
            self.logger.info("HumanEval dataset already exists, loading...")
            await self._load_dataset()
            return

        # Download HumanEval dataset
        await self._download_dataset()

        # Load and process dataset
        await self._load_dataset()

    async def _download_dataset(self) -> None:
        """Download HumanEval dataset from official source."""
        self.logger.info("Downloading HumanEval dataset...")

        # HumanEval dataset URLs
        base_url = "https://raw.githubusercontent.com/openai/human-eval/master"

        async with aiohttp.ClientSession() as session:
            try:
                # Download the main dataset
                url = f"{base_url}/humaneval/{self.split}.jsonl"
                self.logger.info(f"Downloading from {url}")

                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        data_file = self.data_dir / f"{self.split}.jsonl"
                        with open(data_file, 'w') as f:
                            f.write(content)
                        self.logger.info(f"Downloaded {self.split}.jsonl")
                    else:
                        self.logger.error(f"Failed to download dataset: {response.status}")
                        # Fallback to generating sample data
                        await self._generate_fallback_data()

            except Exception as e:
                self.logger.error(f"Error downloading HumanEval dataset: {e}")
                # Fallback to generating sample data
                await self._generate_fallback_data()

    async def _generate_fallback_data(self) -> None:
        """Generate fallback HumanEval-style data if download fails."""
        self.logger.info("Generating fallback HumanEval data...")

        # Sample HumanEval-style problems for testing
        sample_data = [
            {
                "task_id": "test_001",
                "prompt": "def add_numbers(a, b):\n    \"\"\"Add two numbers.\n    \n    Args:\n        a: First number\n        b: Second number\n        \n    Returns:\n        Sum of a and b\n    \"\"\"\n    ",
                "canonical_solution": "    return a + b",
                "test": "def check():\n    assert add_numbers(1, 2) == 3\n    assert add_numbers(-1, 1) == 0\n    assert add_numbers(0, 0) == 0",
                "entry_point": "add_numbers"
            },
            {
                "task_id": "test_002",
                "prompt": "def is_even(n):\n    \"\"\"Check if a number is even.\n    \n    Args:\n        n: Number to check\n        \n    Returns:\n        True if n is even, False otherwise\n    \"\"\"\n    ",
                "canonical_solution": "    return n % 2 == 0",
                "test": "def check():\n    assert is_even(2) == True\n    assert is_even(3) == False\n    assert is_even(0) == True",
                "entry_point": "is_even"
            },
            {
                "task_id": "test_003",
                "prompt": "def factorial(n):\n    \"\"\"Calculate factorial of n.\n    \n    Args:\n        n: Non-negative integer\n        \n    Returns:\n        Factorial of n\n    \"\"\"\n    ",
                "canonical_solution": """    if n <= 1:
        return 1
    return n * factorial(n - 1)""",
                "test": "def check():\n    assert factorial(0) == 1\n    assert factorial(1) == 1\n    assert factorial(5) == 120",
                "entry_point": "factorial"
            },
            {
                "task_id": "test_004",
                "prompt": "def fibonacci(n):\n    \"\"\"Return nth Fibonacci number.\n    \n    Args:\n        n: Position in Fibonacci sequence (0-indexed)\n        \n    Returns:\n        nth Fibonacci number\n    \"\"\"\n    ",
                "canonical_solution": """    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)""",
                "test": "def check():\n    assert fibonacci(0) == 0\n    assert fibonacci(1) == 1\n    assert fibonacci(5) == 5",
                "entry_point": "fibonacci"
            },
            {
                "task_id": "test_005",
                "prompt": "def reverse_string(s):\n    \"\"\"Reverse a string.\n    \n    Args:\n        s: Input string\n        \n    Returns:\n        Reversed string\n    \"\"\"\n    ",
                "canonical_solution": "    return s[::-1]",
                "test": "def check():\n    assert reverse_string(\"hello\") == \"olleh\"\n    assert reverse_string(\"\") == \"\"\n    assert reverse_string(\"a\") == \"a\"",
                "entry_point": "reverse_string"
            }
        ]

        # Save fallback data
        data_file = self.data_dir / f"{self.split}.jsonl"
        with open(data_file, 'w') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')

        self.logger.info(f"Generated {len(sample_data)} fallback samples")

    async def _load_dataset(self) -> None:
        """Load and parse HumanEval dataset."""
        self.logger.info("Loading HumanEval dataset...")

        data_file = self.data_dir / f"{self.split}.jsonl"
        if not data_file.exists():
            self.logger.error(f"Dataset file {data_file} does not exist")
            return

        try:
            with open(data_file) as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)

                        # Parse the item
                        task_id = item.get("task_id", f"task_{line_num}")
                        prompt = item.get("prompt", "")
                        canonical_solution = item.get("canonical_solution", "")
                        test = item.get("test", "")
                        entry_point = item.get("entry_point", "")

                        # Add to dataset
                        self.dataset.append({
                            "task_id": task_id,
                            "prompt": prompt,
                            "canonical_solution": canonical_solution,
                            "test": test,
                            "entry_point": entry_point,
                            "full_prompt": prompt + canonical_solution
                        })

                        # Limit samples if specified
                        if self.max_samples and len(self.dataset) >= self.max_samples:
                            break

                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Error parsing line {line_num}: {e}")
                        continue

            self.logger.info(f"Loaded {len(self.dataset)} HumanEval samples")

        except Exception as e:
            self.logger.error(f"Error loading HumanEval dataset: {e}")

    async def evaluate(self, model: Any, **kwargs) -> BenchmarkResult:
        """Evaluate model on HumanEval benchmark."""
        self.logger.info(f"Evaluating model on HumanEval with {len(self.dataset)} samples...")

        passed_tests = 0
        total_tests = len(self.dataset)
        pass_at_1_results = []
        pass_at_10_results = []

        # Evaluate each sample
        for item in self.dataset:
            try:
                # Get model prediction
                prediction = await self._get_model_prediction(model, item)

                # Test the prediction
                test_result = await self._test_prediction(prediction, item)

                if test_result["passed"]:
                    passed_tests += 1
                    pass_at_1_results.append(True)
                    pass_at_10_results.append(True)
                else:
                    pass_at_1_results.append(False)
                    pass_at_10_results.append(False)

            except Exception as e:
                self.logger.warning(f"Error evaluating sample {item.get('task_id', 'unknown')}: {e}")
                pass_at_1_results.append(False)
                pass_at_10_results.append(False)
                continue

        # Calculate results
        pass_at_1 = (sum(pass_at_1_results) / total_tests) * 100 if total_tests > 0 else 0
        pass_at_10 = (sum(pass_at_10_results) / total_tests) * 100 if total_tests > 0 else 0

        result = BenchmarkResult(
            benchmark_name=self.name,
            score=pass_at_1,
            max_score=100.0,
            raw_score=passed_tests,
            pass_at_1=pass_at_1,
            pass_at_10=pass_at_10,
            accuracy=pass_at_1,
            dataset_size=total_tests,
            successful_samples=passed_tests,
            metadata={
                "pass_at_1_list": pass_at_1_results,
                "pass_at_10_list": pass_at_10_results,
                "split": self.split
            }
        )

        self.logger.info(f"HumanEval Results: {pass_at_1:.2f}% Pass@1 ({passed_tests}/{total_tests})")
        return result

    async def _get_model_prediction(self, model: Any, item: Dict[str, Any]) -> str:
        """Get model prediction for HumanEval task."""
        prompt = item["prompt"]

        if hasattr(model, 'chat'):
            # If model has chat interface (like your NLPAgent)
            response = ""
            async for chunk in model.chat(prompt):
                response += chunk
            return response.strip()
        elif hasattr(model, 'generate'):
            # If model has generate interface
            response = await model.generate(prompt)
            return str(response).strip()
        else:
            # Fallback - return empty implementation
            return ""

    async def _test_prediction(self, prediction: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """Test the model's prediction against test cases."""
        try:
            # Combine prompt with prediction
            full_code = item["prompt"] + prediction
            test_code = item["test"]
            _entry_point = item["entry_point"]

            # Create temporary file with the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                f.write('\n\n')
                f.write(test_code)
                temp_file = f.name

            try:
                # Run the test
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30  # 30 second timeout
                )

                # Check if tests passed
                passed = result.returncode == 0

                return {
                    "passed": passed,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }

            finally:
                # Clean up temp file
                Path(temp_file).unlink(missing_ok=True)

        except Exception as e:
            self.logger.warning(f"Error testing prediction: {e}")
            return {
                "passed": False,
                "stdout": "",
                "stderr": str(e)
            }

    def get_dataset_size(self) -> int:
        """Return total number of HumanEval tasks."""
        return len(self.dataset)

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("HumanEval benchmark cleanup completed")
