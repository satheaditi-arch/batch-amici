"""
BA-AMICI Benchmark Package

A comprehensive benchmark for evaluating Batch-Aware AMICI against baseline AMICI
using interaction-based metrics and cross-replicate consistency.
"""

from .ba_amici_benchmark_pipeline import (
    run_pipeline,
    PipelineConfig,
    DataConfig,
    TrainingConfig,
    EvaluationConfig,
    InteractionConfig,
    generate_benchmark_data,
    train_amici_model,
    evaluate_models,
)

__version__ = "1.0.0"

__all__ = [
    "run_pipeline",
    "PipelineConfig",
    "DataConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "InteractionConfig",
    "generate_benchmark_data",
    "train_amici_model",
    "evaluate_models",
]
