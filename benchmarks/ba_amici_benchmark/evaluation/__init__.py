"""
Evaluation Module

Interaction-based evaluation for AMICI models.
Measures AUPRC for gene/sender/receiver prediction and cross-replicate consistency.
"""

from .interaction_consistency_evaluator import (
    InteractionConsistencyEvaluator,
    InteractionPrediction,
    EvaluationResults
)

__all__ = [
    'InteractionConsistencyEvaluator',
    'InteractionPrediction',
    'EvaluationResults'
]