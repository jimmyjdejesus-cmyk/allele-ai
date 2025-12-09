"""
Genome classes for conversational AI with 8 evolved traits.

This module implements the core genome system for Allele, featuring:
- ConversationalGenome with 8 evolved traits
- Gene expression and regulation
- Mutation and crossover operators
- Fitness evaluation and adaptation

Author: Bravetto AI Systems  
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import numpy as np

from .types import TraitDict, GenomeMetadata, FitnessMetrics
from .exceptions import GenomeError, ValidationError


class Gene:
    """Base gene class for evolutionary algorithms.

    Represents a single unit of genetic material with mutation capabilities.

    Attributes:
        gene_id: Unique identifier for this gene
        name: Human-readable name for the gene
        expression_level: Expression level (0.0 to 1.0)
        mutation_rate: Probability of mutation occurring (0.0 to 1.0)
        regulation_factors: Factors affecting gene regulation
        metadata: Additional metadata for the gene
    """

    def __init__(
        self,
        gene_id: str,
        name: str,
        expression_level: float = 0.5,
        mutation_rate: float = 0.1,
        regulation_factors: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize a gene with comprehensive parameters.

        Args:
            gene_id: Unique identifier for this gene
            name: Human-readable name for the gene
            expression_level: Expression level (0.0 to 1.0, default: 0.5)
            mutation_rate: Probability of mutation (default: 0.1)
            regulation_factors: Dictionary of regulation factors
            metadata: Additional metadata dictionary

        Raises:
            ValidationError: If parameters are invalid
        """
        if not 0.0 <= expression_level <= 1.0:
            raise ValidationError(
                f"Expression level must be between 0.0 and 1.0, got {expression_level}"
            )
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValidationError(
                f"Mutation rate must be between 0.0 and 1.0, got {mutation_rate}"
            )

        self.gene_id: str = gene_id
        self.name: str = name
        self.expression_level: float = expression_level
        self.mutation_rate: float = mutation_rate
        self.regulation_factors: Dict[str, float] = regulation_factors or {}
        self.metadata: Dict[str, Any] = metadata or {}

    def mutate(self, noise_std: float = 0.1) -> None:
        """Apply mutation to this gene's expression level.

        Args:
            noise_std: Standard deviation of Gaussian noise
        """
        noise = np.random.normal(0, noise_std)
        self.expression_level = np.clip(
            self.expression_level + noise, 0.0, 1.0
        )


class GenomeBase(ABC):
    """
    Abstract base class for all genome types.

    All genome types inherit from this to ensure compatibility
    with evolution engines and agent systems.
    """

    def __init__(self, genome_id: str):
        """Initialize genome.

        Args:
            genome_id: Unique identifier
        """
        self.genome_id = genome_id
        self.fitness_history: List[float] = []
        self.generation: int = 0

    @abstractmethod
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        pass

    @abstractmethod
    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenomeBase':
        """Deserialize from dictionary."""
        pass


class ConversationalGenome(GenomeBase):
    """Genome specialized for conversational AI capabilities.

    This genome implements 8 core conversational traits that define the personality
    and behavioral characteristics of an NLP agent. Each trait is encoded as a gene
    with expression levels that can evolve through natural selection.

    Conversational Traits:
        - empathy: Ability to understand and respond to emotional context
        - engagement: Level of conversational engagement and enthusiasm
        - technical_knowledge: Depth of technical understanding and accuracy
        - creativity: Originality and creative problem-solving ability
        - conciseness: Ability to be concise while remaining informative
        - context_awareness: Understanding of conversation history and context
        - adaptability: Ability to adapt to different conversation styles
        - personability: Friendliness and approachability in interactions

    Example:
        >>> genome = ConversationalGenome(
        ...     genome_id="agent_001",
        ...     traits={
        ...         'empathy': 0.9,
        ...         'technical_knowledge': 0.95,
        ...         'creativity': 0.7
        ...     }
        ... )
        >>> genome.get_trait_value('empathy')
        0.9
        >>> genome.mutate_trait('empathy', learning_rate=0.1)
        >>> genome.crossover(other_genome)
    """

    # Default trait values
    DEFAULT_TRAITS: Dict[str, float] = {
        'empathy': 0.5,
        'engagement': 0.5,
        'technical_knowledge': 0.5,
        'creativity': 0.5,
        'conciseness': 0.5,
        'context_awareness': 0.5,
        'adaptability': 0.5,
        'personability': 0.5
    }

    def __init__(
        self,
        genome_id: str,
        traits: Optional[TraitDict] = None,
        metadata: Optional[GenomeMetadata] = None
    ) -> None:
        """Initialize a conversational genome with specific traits.

        Args:
            genome_id: Unique identifier for this genome
            traits: Dictionary mapping trait names to expression levels (0.0-1.0)
            metadata: Optional genome metadata

        Raises:
            ValidationError: If trait values are invalid
        """
        super().__init__(genome_id)

        # Initialize traits with defaults
        self.traits = self.DEFAULT_TRAITS.copy()
        if traits:
            self._validate_traits(traits)
            self.traits.update(traits)

        # Initialize genes
        self.genes: List[Gene] = []
        self._initialize_genes()

        # Metadata
        self.metadata = metadata or GenomeMetadata(
            creation_timestamp=datetime.now(timezone.utc),
            generation=0
        )

        # Sync generation from metadata to base class
        self.generation = self.metadata.generation

        # Fitness tracking
        self.fitness_score: float = 0.0
        self.fitness_metrics: Optional[FitnessMetrics] = None

    def _validate_traits(self, traits: TraitDict) -> None:
        """Validate trait dictionary.

        Args:
            traits: Trait dictionary to validate

        Raises:
            ValidationError: If traits are invalid
        """
        for trait_name, value in traits.items():
            if trait_name not in self.DEFAULT_TRAITS:
                raise ValidationError(
                    f"Unknown trait: {trait_name}. "
                    f"Valid traits: {list(self.DEFAULT_TRAITS.keys())}"
                )
            if not isinstance(value, (int, float)):
                raise ValidationError(
                    f"Trait value must be numeric, got {type(value)} for {trait_name}"
                )
            if not 0.0 <= value <= 1.0:
                raise ValidationError(
                    f"Trait value must be between 0.0 and 1.0, "
                    f"got {value} for {trait_name}"
                )

    def _initialize_genes(self) -> None:
        """Initialize genes for conversational abilities.

        Creates genes for each of the 8 core conversational traits.
        """
        for trait_name, expression_level in self.traits.items():
            gene_id = hashlib.md5(
                f"{self.genome_id}_{trait_name}".encode()
            ).hexdigest()[:8]

            gene = Gene(
                gene_id=gene_id,
                name=f"conversation_{trait_name}",
                expression_level=expression_level,
                mutation_rate=0.1,
                regulation_factors={
                    'stability': 0.8,
                    'plasticity': 0.6,
                    'heritability': 0.9
                },
                metadata={
                    'trait_type': 'conversational',
                    'optimization_target': trait_name
                }
            )
            self.genes.append(gene)

    def get_trait_value(self, trait_name: str, default: float = 0.5) -> float:
        """Get the expression value for a specific conversational trait.

        Args:
            trait_name: Name of the trait to retrieve
            default: Default value if trait not found

        Returns:
            Trait expression level (0.0-1.0)
        """
        return self.traits.get(trait_name, default)

    def set_trait_value(self, trait_name: str, value: float) -> None:
        """Set the expression value for a specific trait.

        Args:
            trait_name: Name of the trait to set
            value: New expression level (0.0-1.0)

        Raises:
            ValidationError: If trait name or value is invalid
        """
        if trait_name not in self.DEFAULT_TRAITS:
            raise ValidationError(f"Unknown trait: {trait_name}")

        if not 0.0 <= value <= 1.0:
            raise ValidationError(
                f"Trait value must be between 0.0 and 1.0, got {value}"
            )

        self.traits[trait_name] = value

        # Update corresponding gene
        for gene in self.genes:
            if gene.metadata.get('optimization_target') == trait_name:
                gene.expression_level = value
                break

    def mutate_trait(
        self,
        trait_name: str,
        mutation_strength: float = 0.1
    ) -> None:
        """Mutate a specific trait.

        Args:
            trait_name: Name of trait to mutate
            mutation_strength: Strength of mutation (standard deviation)

        Raises:
            ValidationError: If trait name is invalid
        """
        if trait_name not in self.traits:
            raise ValidationError(f"Unknown trait: {trait_name}")

        # Apply Gaussian noise
        noise = np.random.normal(0, mutation_strength)
        new_value = np.clip(self.traits[trait_name] + noise, 0.0, 1.0)
        self.set_trait_value(trait_name, new_value)

    def mutate_all_traits(self, mutation_rate: float = 0.1) -> None:
        """Apply mutations to all conversational traits.

        Args:
            mutation_rate: Probability of each trait mutating
        """
        for trait_name in self.traits.keys():
            if np.random.random() < mutation_rate:
                self.mutate_trait(trait_name)

        # Update metadata
        self.metadata.last_mutation = datetime.now(timezone.utc)

    def crossover(self, other: 'ConversationalGenome') -> 'ConversationalGenome':
        """Create offspring through genome crossover.

        Args:
            other: The other genome to crossover with

        Returns:
            A new genome resulting from the crossover operation
        """
        # Blend traits from both parents
        child_traits = {}
        for trait_name in self.traits.keys():
            # Random blend
            if np.random.random() < 0.5:
                child_traits[trait_name] = (
                    0.7 * self.traits[trait_name] +
                    0.3 * other.traits[trait_name]
                )
            else:
                child_traits[trait_name] = (
                    0.3 * self.traits[trait_name] +
                    0.7 * other.traits[trait_name]
                )

            # Add small random variation
            child_traits[trait_name] += np.random.normal(0, 0.05)
            child_traits[trait_name] = np.clip(child_traits[trait_name], 0.0, 1.0)

        # Create child genome
        child_id = f"crossover_{hashlib.md5(str(child_traits).encode()).hexdigest()[:12]}"

        child_metadata = GenomeMetadata(
            creation_timestamp=datetime.now(timezone.utc),
            generation=max(self.generation, other.generation) + 1,
            parent_ids=[self.genome_id, other.genome_id]
        )

        return ConversationalGenome(
            genome_id=child_id,
            traits=child_traits,
            metadata=child_metadata
        )

    def adapt_from_feedback(
        self,
        feedback_score: float,
        learning_rate: float = 0.1
    ) -> None:
        """Adapt genome based on conversational feedback.

        Args:
            feedback_score: Feedback score (0.0-1.0)
            learning_rate: Rate of adaptation (0.0-1.0)

        Raises:
            ValidationError: If parameters are invalid
        """
        if not 0.0 <= feedback_score <= 1.0:
            raise ValidationError(
                f"Feedback score must be between 0.0 and 1.0, got {feedback_score}"
            )

        if not 0.0 <= learning_rate <= 1.0:
            raise ValidationError(
                f"Learning rate must be between 0.0 and 1.0, got {learning_rate}"
            )

        # Strengthen traits that correlate with positive feedback
        for trait_name, current_value in self.traits.items():
            # Calculate adaptation based on feedback
            trait_influence = current_value * (feedback_score - 0.5)
            new_value = current_value + trait_influence * learning_rate
            self.set_trait_value(trait_name, np.clip(new_value, 0.0, 1.0))

    # GenomeBase interface implementation
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get trait value or configuration."""
        if key == 'traits':
            return self.traits
        return self.traits.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """Set trait value."""
        if key in self.traits:
            self.set_trait_value(key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize genome to dictionary.

        Returns:
            Dictionary representation of the genome
        """
        return {
            "genome_id": self.genome_id,
            "traits": self.traits,
            "fitness_score": self.fitness_score,
            "fitness_history": self.fitness_history,
            "generation": self.generation,
            "metadata": {
                "creation_timestamp": self.metadata.creation_timestamp.isoformat(),
                "last_mutation": (
                    self.metadata.last_mutation.isoformat()
                    if self.metadata.last_mutation else None
                ),
                "parent_ids": self.metadata.parent_ids,
                "lineage": self.metadata.lineage,
                "tags": self.metadata.tags
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationalGenome':
        """Deserialize genome from dictionary.

        Args:
            data: Dictionary containing genome data

        Returns:
            ConversationalGenome instance

        Raises:
            ValidationError: If data is invalid
        """
        if "genome_id" not in data:
            raise ValidationError("Missing required field: genome_id")

        # Parse metadata
        metadata_dict = data.get("metadata", {})
        metadata = GenomeMetadata(
            creation_timestamp=datetime.fromisoformat(
                metadata_dict.get("creation_timestamp", datetime.now(timezone.utc).isoformat())
            ),
            last_mutation=(
                datetime.fromisoformat(metadata_dict["last_mutation"])
                if metadata_dict.get("last_mutation") else None
            ),
            generation=data.get("generation", 0),
            parent_ids=metadata_dict.get("parent_ids"),
            lineage=metadata_dict.get("lineage"),
            tags=metadata_dict.get("tags")
        )

        # Create genome
        genome = cls(
            genome_id=data["genome_id"],
            traits=data.get("traits"),
            metadata=metadata
        )

        # Restore fitness info
        genome.fitness_score = data.get("fitness_score", 0.0)
        genome.fitness_history = data.get("fitness_history", [])
        genome.generation = data.get("generation", 0)

        return genome

