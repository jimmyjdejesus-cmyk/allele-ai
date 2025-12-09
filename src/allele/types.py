"""
Type definitions for Allele SDK.

Author: Bravetto AI Systems
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, TypedDict, Literal
from dataclasses import dataclass
from datetime import datetime


# Trait dictionary type
TraitDict = Dict[str, float]

# Trait names (8 core traits)
TraitName = Literal[
    'empathy',
    'engagement', 
    'technical_knowledge',
    'creativity',
    'conciseness',
    'context_awareness',
    'adaptability',
    'personability'
]


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation.
    
    Attributes:
        user_input: The user's message or query
        agent_response: The agent's response
        timestamp: ISO format timestamp of the conversation turn
        context_embedding: Optional vector representation of context
        response_quality_score: Quality score from 0.0 to 1.0
        evolutionary_adaptations: List of adaptations made during this turn
    """
    user_input: str
    agent_response: str
    timestamp: str
    context_embedding: Optional[List[float]] = None
    response_quality_score: float = 0.0
    evolutionary_adaptations: Optional[List[str]] = None


@dataclass
class AgentResponse:
    """Response from an NLP agent.
    
    Attributes:
        content: The response content
        genome_id: ID of the genome that generated the response
        traits_used: Dictionary of trait values used in generation
        quality_score: Estimated quality of the response
        generation_time: Time taken to generate response (seconds)
        metadata: Additional response metadata
    """
    content: str
    genome_id: str
    traits_used: TraitDict
    quality_score: float
    generation_time: float
    metadata: Dict[str, Any]


class AgentConfigDict(TypedDict, total=False):
    """Configuration dictionary for agent creation.
    
    Attributes:
        model_name: Name of the LLM model to use
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        streaming: Whether to enable streaming responses
        memory_enabled: Whether to enable conversation memory
        evolution_enabled: Whether to enable evolutionary adaptation
    """
    model_name: str
    temperature: float
    max_tokens: int
    streaming: bool
    memory_enabled: bool
    evolution_enabled: bool


class EvolutionConfigDict(TypedDict, total=False):
    """Configuration dictionary for evolution engine.
    
    Attributes:
        population_size: Number of genomes in population
        generations: Maximum number of generations to evolve
        mutation_rate: Probability of gene mutation (0.0-1.0)
        crossover_rate: Probability of parent crossover (0.0-1.0)
        selection_pressure: Fraction of best individuals to keep (0.0-1.0)
        elitism_enabled: Whether to preserve best genomes
    """
    population_size: int
    generations: int
    mutation_rate: float
    crossover_rate: float
    selection_pressure: float
    elitism_enabled: bool


@dataclass
class FitnessMetrics:
    """Fitness metrics for genome evaluation.
    
    Attributes:
        relevance: Relevance score (0.0-1.0)
        coherence: Coherence score (0.0-1.0)
        engagement: Engagement score (0.0-1.0)
        helpfulness: Helpfulness score (0.0-1.0)
        technical_accuracy: Technical accuracy score (0.0-1.0)
        creativity: Creativity score (0.0-1.0)
        empathy: Empathy score (0.0-1.0)
        overall_fitness: Overall fitness score (0.0-1.0)
    """
    relevance: float
    coherence: float
    engagement: float
    helpfulness: float
    technical_accuracy: float
    creativity: float
    empathy: float
    overall_fitness: float


@dataclass
class GenomeMetadata:
    """Metadata for genome instances.
    
    Attributes:
        creation_timestamp: When the genome was created
        last_mutation: When the genome was last mutated
        generation: Generation number in evolution
        parent_ids: IDs of parent genomes (for crossover)
        lineage: Complete lineage trace
        tags: User-defined tags for organization
    """
    creation_timestamp: datetime
    last_mutation: Optional[datetime] = None
    generation: int = 0
    parent_ids: Optional[List[str]] = None
    lineage: Optional[List[str]] = None
    tags: Optional[List[str]] = None

