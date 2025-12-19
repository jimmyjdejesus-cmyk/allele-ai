"""
MMLU (Massive Multitask Language Understanding) Benchmark Implementation.

This module implements the MMLU benchmark for evaluating knowledge-based reasoning
across 57 academic subjects including STEM, humanities, and social sciences.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import aiohttp
import zipfile

from .base import Benchmark, BenchmarkResult
from .registry import register_benchmark


@register_benchmark("mmlu")
class MMLUBenchmark(Benchmark):
    """
    MMLU Benchmark for Massive Multitask Language Understanding.
    
    Evaluates knowledge-based reasoning across 57 academic subjects including:
    - STEM subjects (mathematics, physics, chemistry, biology)
    - Humanities (history, literature, philosophy, art)
    - Social sciences (economics, psychology, sociology, politics)
    - Professional fields (law, medicine, engineering, business)
    
    Each question is multiple choice with 4 options (A, B, C, D).
    """
    
    def __init__(self, subjects: Optional[List[str]] = None, max_samples: Optional[int] = None):
        """
        Initialize MMLU benchmark.
        
        Args:
            subjects: List of subjects to test (None for all subjects)
            max_samples: Maximum number of samples to evaluate (None for all)
        """
        description = "Massive Multitask Language Understanding across 57 academic subjects"
        super().__init__("MMLU", description, max_score=100.0)
        
        self.subjects = subjects
        self.max_samples = max_samples
        self.data_dir = Path("benchmarks/data/mmlu")
        self.dataset = []
        self.subject_list = []
        
    async def setup(self) -> None:
        """Download and prepare MMLU dataset."""
        self.logger.info("Setting up MMLU benchmark...")
        
        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if dataset already exists
        if self.data_dir.exists() and (self.data_dir / "dev.json").exists():
            self.logger.info("MMLU dataset already exists, loading...")
            await self._load_dataset()
            return
        
        # Download MMLU dataset
        await self._download_dataset()
        
        # Load and process dataset
        await self._load_dataset()
        
    async def _download_dataset(self) -> None:
        """Download MMLU dataset from official source."""
        self.logger.info("Downloading MMLU dataset...")
        
        # MMLU dataset URLs
        base_url = "https://github.com/hendrycks/test/raw/master/data"
        
        async with aiohttp.ClientSession() as session:
            # Download subject list first
            try:
                async with session.get(f"{base_url}/test.csv") as response:
                    content = await response.text()
                    
                    # Extract subject names from test.csv
                    lines = content.strip().split('\n')
                    if lines:
                        # Parse CSV to get unique subjects
                        self.subject_list = sorted(set(line.split(',')[0] for line in lines[1:]))
                        
                        self.logger.info(f"Found {len(self.subject_list)} subjects: {self.subject_list[:10]}...")
                        
                        # Save subject list
                        with open(self.data_dir / "subjects.txt", 'w') as f:
                            for subject in self.subject_list:
                                f.write(f"{subject}\n")
                        
            except Exception as e:
                self.logger.error(f"Error downloading subject list: {e}")
                # Fallback to known MMLU subjects
                self.subject_list = self._get_fallback_subjects()
            
            # Download dev split for each subject
            tasks = []
            for subject in (self.subjects or self.subject_list)[:10]:  # Limit for initial implementation
                task = self._download_subject_data(session, base_url, subject)
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _get_fallback_subjects(self) -> List[str]:
        """Fallback list of MMLU subjects if download fails."""
        return [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
            "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
            "college_medicine", "college_physics", "computer_security", "conceptual_physics",
            "electrical_engineering", "elementary_mathematics", "high_school_biology",
            "high_school_chemistry", "high_school_computer_science", "high_school_mathematics",
            "high_school_physics", "high_school_statistics", "human_sexuality", "international_law",
            "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing",
            "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
            "philosophy", "prehistory", "professional_accounting", "professional_medicine",
            "professional_psychology", "public_relations", "security_studies", "sociology",
            "us_foreign_policy", "virology", "world_history"
        ]
    
    async def _download_subject_data(self, session: aiohttp.ClientSession, base_url: str, subject: str) -> None:
        """Download data for a specific subject."""
        try:
            # Try multiple file formats
            for filename in [f"{subject}_test.csv", f"{subject}_dev.csv"]:
                url = f"{base_url}/{filename}"
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        filepath = self.data_dir / filename
                        with open(filepath, 'w') as f:
                            f.write(content)
                        self.logger.info(f"Downloaded {filename}")
                        break
        except Exception as e:
            self.logger.warning(f"Error downloading {subject}: {e}")
    
    async def _load_dataset(self) -> None:
        """Load and parse MMLU dataset."""
        self.logger.info("Loading MMLU dataset...")
        
        # Load subject list
        if (self.data_dir / "subjects.txt").exists():
            with open(self.data_dir / "subjects.txt", 'r') as f:
                self.subject_list = [line.strip() for line in f if line.strip()]
        
        # Load all subject data files
        for subject in (self.subjects or self.subject_list):
            subject_file = self.data_dir / f"{subject}_test.csv"
            if subject_file.exists():
                await self._parse_subject_file(subject, subject_file)
            else:
                # Try dev file
                dev_file = self.data_dir / f"{subject}_dev.csv"
                if dev_file.exists():
                    await self._parse_subject_file(subject, dev_file)
    
    async def _parse_subject_file(self, subject: str, filepath: Path) -> None:
        """Parse a subject's data file."""
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Parse CSV format: question, A, B, C, D, answer
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(',')
                if len(parts) >= 6:
                    question = parts[0]
                    options = parts[1:5]
                    correct_answer = parts[5]
                    
                    # Add to dataset
                    self.dataset.append({
                        "subject": subject,
                        "question": question,
                        "options": options,
                        "correct_answer": correct_answer,
                        "choices": ["A", "B", "C", "D"][:len(options)]
                    })
                    
                    # Limit samples if specified
                    if self.max_samples and len(self.dataset) >= self.max_samples:
                        break
            
            self.logger.info(f"Loaded {len([d for d in self.dataset if d['subject'] == subject])} samples for {subject}")
                        
        except Exception as e:
            self.logger.error(f"Error parsing {filepath}: {e}")
    
    async def evaluate(self, model: Any, **kwargs) -> BenchmarkResult:
        """Evaluate model on MMLU benchmark."""
        self.logger.info(f"Evaluating model on MMLU with {len(self.dataset)} samples...")
        
        correct_count = 0
        total_count = len(self.dataset)
        subject_scores = {}
        
        # Group by subject for analysis
        subject_data = {}
        for item in self.dataset:
            subject = item["subject"]
            if subject not in subject_data:
                subject_data[subject] = []
            subject_data[subject].append(item)
        
        # Evaluate each sample
        for item in self.dataset:
            try:
                # Create prompt
                prompt = self._create_prompt(item)
                
                # Get model prediction
                prediction = await self._get_model_prediction(model, prompt)
                
                # Check if correct
                is_correct = self._check_answer(prediction, item["correct_answer"])
                if is_correct:
                    correct_count += 1
                
                # Track by subject
                subject = item["subject"]
                if subject not in subject_scores:
                    subject_scores[subject] = {"correct": 0, "total": 0}
                subject_scores[subject]["total"] += 1
                if is_correct:
                    subject_scores[subject]["correct"] += 1
                
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
                "subject_scores": subject_scores,
                "total_subjects": len(subject_data),
                "average_per_subject": accuracy
            }
        )
        
        self.logger.info(f"MMLU Results: {accuracy:.2f}% ({correct_count}/{total_count})")
        return result
    
    def _create_prompt(self, item: Dict[str, Any]) -> str:
        """Create MMLU prompt for a question."""
        question = item["question"]
        options = item["options"]
        
        prompt = f"Question: {question}\n"
        prompt += "Options:\n"
        for i, option in enumerate(options):
            letter = chr(65 + i)  # A, B, C, D
            prompt += f"{letter}. {option}\n"
        prompt += "Answer:"
        
        return prompt
    
    async def _get_model_prediction(self, model: Any, prompt: str) -> str:
        """Get model prediction for MMLU question."""
        # This is a placeholder - implementation depends on the model interface
        # For now, we'll use a simple approach that can be adapted
        
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
            # Fallback - this should be customized based on your model
            return "A"
    
    def _check_answer(self, prediction: str, correct_answer: str) -> bool:
        """Check if model prediction is correct."""
        # Extract letter from prediction
        prediction = prediction.upper().strip()
        correct_answer = correct_answer.upper().strip()
        
        # Check if prediction contains the correct answer
        return correct_answer in prediction
    
    def get_dataset_size(self) -> int:
        """Return total number of MMLU questions."""
        return len(self.dataset)
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("MMLU benchmark cleanup completed")
