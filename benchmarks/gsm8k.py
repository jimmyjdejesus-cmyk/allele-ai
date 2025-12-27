"""
GSM8K Benchmark Implementation.

DEPRECATED: This custom implementation is deprecated in favor of the official lm-eval harness.
Please use `scripts/run_lm_eval_mass.py` for standard benchmarking.

This module implements the GSM8K benchmark for evaluating mathematical reasoning
through grade school math word problems.
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import aiohttp

from .base import Benchmark, BenchmarkResult
from .registry import register_benchmark


@register_benchmark("gsm8k")
class GSM8KBenchmark(Benchmark):
    """
    GSM8K Benchmark for Mathematical Reasoning.
    
    Evaluates mathematical reasoning through grade school math word problems.
    Each problem requires multi-step reasoning to reach a numerical answer.
    
    Format:
    - Question: Math word problem
    - Answer: Numerical answer (integer)
    """
    
    def __init__(self, split: str = "test", max_samples: Optional[int] = None):
        """
        Initialize GSM8K benchmark.
        
        Args:
            split: Dataset split ('train', 'test')
            max_samples: Maximum number of samples to evaluate
        """
        description = "GSM8K: Grade school mathematics word problems"
        super().__init__("GSM8K", description, max_score=100.0)
        
        self.split = split
        self.max_samples = max_samples
        self.data_dir = Path("benchmarks/data/gsm8k")
        self.dataset = []
        
    async def setup(self) -> None:
        """Download and prepare GSM8K dataset."""
        self.logger.info("Setting up GSM8K benchmark...")
        
        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if dataset already exists
        data_file = self.data_dir / f"{self.split}.jsonl"
        if data_file.exists():
            self.logger.info("GSM8K dataset already exists, loading...")
            await self._load_dataset()
            return
        
        # Download GSM8K dataset
        await self._download_dataset()
        
        # Load and process dataset
        await self._load_dataset()
        
    async def _download_dataset(self) -> None:
        """Download GSM8K dataset from official source."""
        self.logger.info("Downloading GSM8K dataset...")
        
        # GSM8K dataset URLs
        base_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/data"
        
        async with aiohttp.ClientSession() as session:
            try:
                # Download the main dataset
                url = f"{base_url}/{self.split}.jsonl"
                self.logger.info(f"Downloading from {url}")
                
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        data_file = self.data_dir / f"{self.split}.jsonl"
                        with open(data_file, 'w') as f:
                            f.write(content)
                        self.logger.info(f"Downloaded {self.split}.json:
                        self.loggerl")
                    else.error(f"Failed to download dataset: {response.status}")
                        # Fallback to generating sample data
                        await self._generate_fallback_data()
                        
            except Exception as e:
                self.logger.error(f"Error downloading GSM8K dataset: {e}")
                # Fallback to generating sample data
                await self._generate_fallback_data()
    
    async def _generate_fallback_data(self) -> None:
        """Generate fallback GSM8K-style data if download fails."""
        self.logger.info("Generating fallback GSM8K data...")
        
        # Sample GSM8K-style problems for testing
        sample_data = [
            {
                "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. How many eggs does Janet have left at the end of the day?",
                "answer": "9"
            },
            {
                "question": "A robe takes 2 bolts of blue fiber and half as much white fiber. It takes 2/3 bolts of blue fiber. How many bolts in total does it take?",
                "answer": "3"
            },
            {
                "question": "Josh decides to make pancakes for breakfast. He has 12 eggs and each pancake needs 2 eggs. How many pancakes can he make?",
                "answer": "6"
            },
            {
                "question": "Sam has 4 bags of marbles. Each bag has 6 marbles. His friend gives him 3 more bags with 5 marbles each. How many marbles does Sam have total?",
                "answer": "39"
            },
            {
                "question": "A company has 120 employees. If 3/4 of them work from home, how many employees work in the office?",
                "answer": "30"
            },
            {
                "question": "Tom bought 3 packs of gum. Each pack has 5 pieces. He gave away 7 pieces to his friends. How many pieces does he have left?",
                "answer": "8"
            },
            {
                "question": "A garden has 24 plants. If 1/3 are roses and 1/4 are tulips, how many plants are neither roses nor tulips?",
                "answer": "10"
            },
            {
                "question": "Sarah has 18 cookies. She wants to share them equally among 3 friends. How many cookies will each friend get?",
                "answer": "6"
            },
            {
                "question": "A bookshelf has 5 shelves. Each shelf holds 8 books. If 3 shelves are full and 2 are half full, how many books are on the bookshelf?",
                "answer": "28"
            },
            {
                "question": "Mike ran 2.5 miles on Monday, 3 miles on Wednesday, and 1.5 miles on Friday. How many total miles did he run?",
                "answer": "7"
            }
        ]
        
        # Save fallback data
        data_file = self.data_dir / f"{self.split}.jsonl"
        with open(data_file, 'w') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
        
        self.logger.info(f"Generated {len(sample_data)} fallback samples")
    
    async def _load_dataset(self) -> None:
        """Load and parse GSM8K dataset."""
        self.logger.info("Loading GSM8K dataset...")
        
        data_file = self.data_dir / f"{self.split}.jsonl"
        if not data_file.exists():
            self.logger.error(f"Dataset file {data_file} does not exist")
            return
        
        try:
            with open(data_file, 'r') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        item = json.loads(line)
                        
                        # Parse the item
                        question = item.get("question", "")
                        answer = item.get("answer", "")
                        
                        # Add to dataset
                        self.dataset.append({
                            "question": question,
                            "answer": answer
                        })
                        
                        # Limit samples if specified
                        if self.max_samples and len(self.dataset) >= self.max_samples:
                            break
                            
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Error parsing line {line_num}: {e}")
                        continue
            
            self.logger.info(f"Loaded {len(self.dataset)} GSM8K samples")
                        
        except Exception as e:
            self.logger.error(f"Error loading GSM8K dataset: {e}")
    
    async def evaluate(self, model: Any, **kwargs) -> BenchmarkResult:
        """Evaluate model on GSM8K benchmark."""
        self.logger.info(f"Evaluating model on GSM8K with {len(self.dataset)} samples...")
        
        correct_count = 0
        total_count = len(self.dataset)
        step_by_step_scores = []
        
        # Evaluate each sample
        for item in self.dataset:
            try:
                # Create prompt
                prompt = self._create_prompt(item)
                
                # Get model prediction
                prediction = await self._get_model_prediction(model, prompt)
                
                # Check if correct
                is_correct = self._check_answer(prediction, item["answer"])
                
                if is_correct:
                    correct_count += 1
                
                # For analysis - check if reasoning is step-by-step
                has_reasoning = self._analyze_reasoning(prediction)
                step_by_step_scores.append(has_reasoning)
                
            except Exception as e:
                self.logger.warning(f"Error evaluating sample: {e}")
                continue
        
        # Calculate results
        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
        reasoning_score = (sum(step_by_step_scores) / len(step_by_step_scores)) * 100 if step_by_step_scores else 0
        
        result = BenchmarkResult(
            benchmark_name=self.name,
            score=accuracy,
            max_score=100.0,
            raw_score=correct_count,
            accuracy=accuracy,
            dataset_size=total_count,
            successful_samples=correct_count,
            metadata={
                "reasoning_score": reasoning_score,
                "step_by_step_analysis": step_by_step_scores,
                "split": self.split
            }
        )
        
        self.logger.info(f"GSM8K Results: {accuracy:.2f}% ({correct_count}/{total_count})")
        return result
    
    def _create_prompt(self, item: Dict[str, Any]) -> str:
        """Create GSM8K prompt for a question."""
        question = item["question"]
        
        prompt = f"Question: {question}\n"
        prompt += "Answer:"
        
        return prompt
    
    async def _get_model_prediction(self, model: Any, prompt: str) -> str:
        """Get model prediction for GSM8K question."""
        # This is a placeholder - implementation depends on the model interface
        
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
            # Fallback - return empty response
            return ""
    
    def _check_answer(self, prediction: str, correct_answer: str) -> bool:
        """Check if model prediction is correct."""
        # Extract numerical answer from prediction
        predicted_number = self._extract_number(prediction)
        correct_number = self._extract_number(correct_answer)
        
        # Compare as floats to handle different formats
        try:
            return abs(float(predicted_number) - float(correct_number)) < 0.001
        except (ValueError, TypeError):
            return False
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract the final numerical answer from text."""
        # Look for numbers at the end of the text
        # Pattern to match numbers (including decimals)
        number_pattern = r'-?\d+(?:\.\d+)?'
        
        # Find all numbers in the text
        numbers = re.findall(number_pattern, text)
        
        if numbers:
            # Return the last number found
            return numbers[-1]
        
        return None
    
    def _analyze_reasoning(self, prediction: str) -> bool:
        """Analyze if the prediction shows step-by-step reasoning."""
        # Look for reasoning indicators
        reasoning_indicators = [
            "step", "first", "then", "next", "after", "before", "finally",
            "let me", "calculate", "compute", "solve", "answer is",
            "therefore", "so", "which means", "this gives"
        ]
        
        prediction_lower = prediction.lower()
        
        # Check if any reasoning indicators are present
        for indicator in reasoning_indicators:
            if indicator in prediction_lower:
                return True
        
        # Check if multiple sentences with calculations
        sentences = prediction.split('.')
        calculation_words = ['+', '-', '*', '/', '=', '×', '÷']
        
        for sentence in sentences:
            if any(word in sentence for word in calculation_words):
                return True
        
        return False
    
    def get_dataset_size(self) -> int:
        """Return total number of GSM8K problems."""
        return len(self.dataset)
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("GSM8K benchmark cleanup completed")
