"""
HellaSwag Benchmark Implementation.

DEPRECATED: This custom implementation is deprecated in favor of the official lm-eval harness.
Please use `scripts/run_lm_eval_mass.py` for standard benchmarking.

This module implements the HellaSwag benchmark for evaluating commonsense reasoning
through adversarial sentence completion.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from phylogenic.benchmark.utils import check_answer

from .base import Benchmark, BenchmarkResult
from .registry import register_benchmark


@register_benchmark("hellaswag")
class HellaSwagBenchmark(Benchmark):
    """
    HellaSwag Benchmark for Commonsense Reasoning.

    Evaluates commonsense reasoning through adversarial sentence completion.
    Given a context and 4 options, models must choose the most plausible ending.

    Each example contains:
    - Context: A sentence or paragraph
    - 4 options: Possible completions
    - Correct answer: The most plausible ending
    """

    def __init__(self, split: str = "val", max_samples: Optional[int] = None):
        """
        Initialize HellaSwag benchmark.

        Args:
            split: Dataset split ('train', 'val', 'test')
            max_samples: Maximum number of samples to evaluate
        """
        description = "HellaSwag: Commonsense reasoning through adversarial sentence completion"
        super().__init__("HellaSwag", description, max_score=100.0)

        self.split = split
        self.max_samples = max_samples
        self.data_dir = Path("benchmarks/data/hellaswag")
        self.dataset = []

    async def setup(self) -> None:
        """Download and prepare HellaSwag dataset."""
        self.logger.info("Setting up HellaSwag benchmark...")

        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Check if dataset already exists
        data_file = self.data_dir / f"{self.split}.jsonl"
        if data_file.exists():
            self.logger.info("HellaSwag dataset already exists, loading...")
            await self._load_dataset()
            return

        # Download HellaSwag dataset
        await self._download_dataset()

        # Load and process dataset
        await self._load_dataset()

    async def _download_dataset(self) -> None:
        """Download HellaSwag dataset from official source."""
        self.logger.info("Downloading HellaSwag dataset...")

        # HellaSwag dataset URLs
        base_url = "https://github.com/rowanz/hellaswag/raw/master/data"

        async with aiohttp.ClientSession() as session:
            try:
                url = f"{base_url}/{self.split}.jsonl"
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
                self.logger.error(f"Error downloading HellaSwag dataset: {e}")
                # Fallback to generating sample data
                await self._generate_fallback_data()

    async def _generate_fallback_data(self) -> None:
        """Generate fallback HellaSwag-style data if download fails."""
        self.logger.info("Generating fallback HellaSwag data...")

        # Sample HellaSwag-style questions for testing
        sample_data = [
            {
                "ctx": "A person is at a coffee shop ordering a drink.",
                "endings": [
                    "They order a glass of water because they're thirsty.",
                    "They order a coffee because they need caffeine.",
                    "They order a salad because they're hungry.",
                    "They order nothing and just sit down."
                ],
                "label": 1,  # Index of correct answer
                "split": self.split
            },
            {
                "ctx": "A person is walking their dog in the park.",
                "endings": [
                    "The dog starts chasing squirrels.",
                    "The dog starts reading a newspaper.",
                    "The dog starts flying in the sky.",
                    "The dog starts cooking dinner."
                ],
                "label": 0,  # Index of correct answer
                "split": self.split
            },
            {
                "ctx": "Someone is giving a presentation to a group.",
                "endings": [
                    "They use slides and speak clearly to the audience.",
                    "They run away from the room screaming.",
                    "They start singing opera loudly.",
                    "They hide under the table."
                ],
                "label": 0,  # Index of correct answer
                "split": self.split
            },
            {
                "ctx": "A person is cooking dinner in the kitchen.",
                "endings": [
                    "They chop vegetables and stir the sauce.",
                    "They throw the ingredients at the wall.",
                    "They eat the raw ingredients directly.",
                    "They use the food to paint a picture."
                ],
                "label": 0,  # Index of correct answer
                "split": self.split
            },
            {
                "ctx": "Someone is trying to start their car in the morning.",
                "endings": [
                    "They put the key in the ignition and turn it.",
                    "They sing a song to the car.",
                    "They try to push the car to start it.",
                    "They ask the car politely to start."
                ],
                "label": 0,  # Index of correct answer
                "split": self.split
            }
        ]

        # Save fallback data
        data_file = self.data_dir / f"{self.split}.jsonl"
        with open(data_file, 'w') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')

        self.logger.info(f"Generated {len(sample_data)} fallback samples")

    async def _load_dataset(self) -> None:
        """Load and parse HellaSwag dataset."""
        self.logger.info("Loading HellaSwag dataset...")

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
                        ctx = item.get("ctx", "")
                        endings = item.get("endings", [])
                        label = item.get("label", 0)

                        # Add to dataset
                        self.dataset.append({
                            "context": ctx,
                            "options": endings,
                            "correct_answer": label,
                            "choices": ["A", "B", "C", "D"][:len(endings)]
                        })

                        # Limit samples if specified
                        if self.max_samples and len(self.dataset) >= self.max_samples:
                            break

                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Error parsing line {line_num}: {e}")
                        continue

            self.logger.info(f"Loaded {len(self.dataset)} HellaSwag samples")

        except Exception as e:
            self.logger.error(f"Error loading HellaSwag dataset: {e}")

    async def evaluate(self, model: Any, **kwargs) -> BenchmarkResult:
        """Evaluate model on HellaSwag benchmark."""
        self.logger.info(f"Evaluating model on HellaSwag with {len(self.dataset)} samples...")

        correct_count = 0
        total_count = len(self.dataset)
        option_scores = {"A": 0, "B": 0, "C": 0, "D": 0}

        # Evaluate each sample
        for item in self.dataset:
            try:
                # Create prompt
                prompt = self._create_prompt(item)

                # Get model prediction
                prediction = await self._get_model_prediction(model, prompt, item["options"])

# Check if correct using central utility
                correct_option = item["choices"][item["correct_answer"]]
                if check_answer(prediction, correct_option):
                    correct_count += 1

                # Track option selection
                if prediction in option_scores:
                    option_scores[prediction] += 1

            except Exception as e:
                self.logger.warning(f"Error evaluating sample: {e}")
                continue

        # Calculate results
        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

        result = BenchmarkResult(
            benchmark_name=self.name,
            score=accuracy,
            max_score=100.0,
            raw_score=correct_count,
            accuracy=accuracy,
            dataset_size=total_count,
            successful_samples=correct_count,
            metadata={
                "option_selection": option_scores,
                "split": self.split
            }
        )

        self.logger.info(f"HellaSwag Results: {accuracy:.2f}% ({correct_count}/{total_count})")
        return result

    def _create_prompt(self, item: Dict[str, Any]) -> str:
        """Create HellaSwag prompt for a question."""
        context = item["context"]
        options = item["options"]

        prompt = f"Context: {context}\n"
        prompt += "Options:\n"
        for i, option in enumerate(options):
            letter = chr(65 + i)  # A, B, C, D
            prompt += f"{letter}. {option}\n"
        prompt += "Most plausible ending:"

        return prompt

    async def _get_model_prediction(self, model: Any, prompt: str, options: List[str]) -> str:
        """Get model prediction for HellaSwag question."""
        # This is a placeholder - implementation depends on the model interface

        if hasattr(model, 'chat'):
            # If model has chat interface (like your NLPAgent)
            response = ""
            async for chunk in model.chat(prompt):
                response += chunk
            return self._extract_option(response, options)
        elif hasattr(model, 'generate'):
            # If model has generate interface
            response = await model.generate(prompt)
            return self._extract_option(str(response), options)
        else:
            # Fallback - random selection for testing
            import random
            return chr(65 + random.randint(0, len(options) - 1))

    def _extract_option(self, response: str, options: List[str]) -> str:
        """Extract option letter from model response."""
        response = response.upper().strip()

        # Look for option letters (A, B, C, D)
        for i, _option in enumerate(options):
            letter = chr(65 + i)
            if letter in response:
                return letter

        # If no letter found, try to match option text
        for i, option in enumerate(options):
            letter = chr(65 + i)
            if option.lower() in response.lower():
                return letter

        # Default to first option
        return "A" if options else "A"

    # _check_answer removed in favor of central `check_answer` utility

    def get_dataset_size(self) -> int:
        """Return total number of HellaSwag questions."""
        return len(self.dataset)

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("HellaSwag benchmark cleanup completed")
