"""
BA-AMICI Benchmark Pipeline

Complete pipeline for benchmarking Batch-Aware AMICI against baseline AMICI.
Integrates semi-synthetic data generation, model training, and interaction
consistency evaluation.

Scientific Approach:
1. Generate semi-synthetic data with biological subclusters (not artificial scaling)
2. Train baseline AMICI and BA-AMICI on same data
3. Evaluate using interaction-based metrics (AUPRC) not embedding metrics (iLISI)
4. Compare cross-replicate consistency of detected interactions

"""

import os
import sys
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import precision_recall_curve, auc
from scvi.data import AnnDataManager
from scvi.data import fields
from scvi.dataloaders import AnnDataLoader
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))


# Import from our modules
from .data_generation.semisynthetic_batch_generator import (
    SemisyntheticBatchGenerator,
    InteractionRule,
    BatchEffectConfig,
    GroundTruth,
    sample_batch_configs
)
from .evaluation.interaction_consistency_evaluator import (
    InteractionConsistencyEvaluator,
    InteractionPrediction,
    EvaluationResults
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for data generation."""
    source_data_path: str = "data/pbmc_68k.h5ad"
    n_cell_types: int = 3
    n_subclusters_per_type: int = 3
    n_hvgs: int = 500
    use_scvi: bool = True
    n_cells_per_replicate: int = 20000
    n_batches_per_replicate: int = 3
    n_replicates: int = 10
    spatial_width: float = 2000.0
    spatial_height: float = 1000.0
    test_slice_start: float = 900.0
    test_slice_end: float = 1100.0
    
    # Batch effect parameters
    library_size_mean: float = 1.0
    library_size_std: float = 0.5
    dropout_increase: float = 0.12
    gene_noise_std: float = 0.8


@dataclass
class InteractionConfig:
    """Configuration for ground-truth interactions."""
    interactions: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "A_to_B": {
            "sender_type": "A",
            "receiver_type": "B",
            "length_scale": 10.0,
            "expected_genes": 50
        },
        "C_to_A": {
            "sender_type": "C",
            "receiver_type": "A",
            "length_scale": 20.0,
            "expected_genes": 30
        }
    })


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    n_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    n_attention_heads: int = 4
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.1
    
    # BA-AMICI specific
    adversarial_weight: float = 0.1
    batch_embedding_dim: int = 16
    gradient_reversal_lambda: float = 1.0
    
    # Training settings
    early_stopping_patience: int = 10
    checkpoint_frequency: int = 10
    device: str = "cuda"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    top_k_genes: int = 100
    top_k_interactions: int = 100
    significance_threshold: float = 0.05
    min_lfc: float = 0.2
    compute_statistical_tests: bool = True
    generate_visualizations: bool = True


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    interactions: InteractionConfig = field(default_factory=InteractionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output_dir: str = "results/ba_amici_benchmark"
    random_seed: int = 42
    data_only: bool = True


    @classmethod
    def from_yaml(cls, path: str) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            interactions=InteractionConfig(**config_dict.get('interactions', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            output_dir=config_dict.get('output_dir', 'results/ba_amici_benchmark'),
            random_seed=config_dict.get('random_seed', 42)
        )
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'data': asdict(self.data),
            'interactions': asdict(self.interactions),
            'training': asdict(self.training),
            'evaluation': asdict(self.evaluation),
            'output_dir': self.output_dir,
            'random_seed': self.random_seed
        }
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# =============================================================================
# Data Generation
# =============================================================================

def load_source_data(config: DataConfig) -> ad.AnnData:
    """
    Load source single-cell data for semi-synthetic generation.
    
    Parameters
    ----------
    config : DataConfig
        Data configuration
        
    Returns
    -------
    ad.AnnData
        Source data for sampling
    """
    logger.info(f"Loading source data from {config.source_data_path}")
    
    if os.path.exists(config.source_data_path):
        adata = sc.read_h5ad(config.source_data_path)
    else:
        # Fallback to scanpy datasets for testing
        logger.warning(f"Source data not found at {config.source_data_path}")
        logger.info("Downloading PBMC3k dataset as fallback")
        adata = sc.datasets.pbmc3k()
        
        # Basic preprocessing
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
        adata = adata[adata.obs['pct_counts_mt'] < 20].copy()
    
    logger.info(f"Loaded {adata.n_obs} cells with {adata.n_vars} genes")
    return adata


def generate_benchmark_data(
    config: PipelineConfig,
    output_dir: str
) -> Tuple[List[ad.AnnData], List[GroundTruth]]:
    """
    Generate semi-synthetic benchmark data with biological subclusters.
    
    This follows the AMICI paper methodology (Section 4.8.1):
    1. Cluster source data into cell types
    2. Subcluster each cell type to get natural variation
    3. Assign interacting vs neutral subtypes based on spatial proximity
    4. Sample expression from appropriate subclusters
    5. Apply realistic batch effects
    
    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration
    output_dir : str
        Directory to save generated data
        
    Returns
    -------
    Tuple[List[ad.AnnData], List[GroundTruth]]
        List of replicate AnnData objects and their ground truths
    """
    logger.info("="*60)
    logger.info("GENERATING SEMI-SYNTHETIC BENCHMARK DATA")
    logger.info("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load source data
    source_adata = load_source_data(config.data)
    if source_adata.n_obs > 20000:
        logger.info(f"Subsampling from {source_adata.n_obs} to 20000 cells")
        sc.pp.subsample(source_adata, n_obs=20000, random_state=config.random_seed)
        logger.info(f"After subsampling: {source_adata.n_obs} cells")

    # Initialize generator
    generator = SemisyntheticBatchGenerator(
        source_adata=source_adata,
        n_cell_types=config.data.n_cell_types,
        n_subclusters_per_type=config.data.n_subclusters_per_type,
        seed=config.random_seed
    )
    
    # Prepare source (clustering, subclustering)
    logger.info("Preparing source data...")
    generator.prepare_source(
        n_hvgs=config.data.n_hvgs,
        use_scvi=config.data.use_scvi
    )
    
    # Subcluster cell types
    logger.info("Subclustering cell types...")
    generator.subcluster_cell_types()
    
    # Build interactions dict from config
    interactions = {}
    for name, params in config.interactions.interactions.items():
        interactions[name] = {
            "sender": params['sender_type'],
            "receiver": params['receiver_type'],
            "length_scale": params['length_scale'],
        }

     # Generate replicates
    replicates = []
    ground_truths = []

    for rep_idx in range(config.data.n_replicates):
        logger.info(f"\nGenerating replicate {rep_idx + 1}/{config.data.n_replicates}")
        
        # Use different seed for each replicate
        rep_seed = config.random_seed + rep_idx * 1000
        np.random.seed(rep_seed)
        
        # Generate RANDOM batch configs for this replicate
        rep_batch_configs = sample_batch_configs(
            n_batches=config.data.n_batches_per_replicate,
            library_size_mean=config.data.library_size_mean,
            library_size_std=config.data.library_size_std,
            dropout_mean=config.data.dropout_increase,
            dropout_std=config.data.dropout_increase * 0.3,
            noise_mean=config.data.gene_noise_std,
            noise_std=config.data.gene_noise_std * 0.3,
            seed=rep_seed,
        )
        
        # Generate replicate
        adata, ground_truth = generator.generate(
            interactions=interactions,
            n_cells=config.data.n_cells_per_replicate,
            n_batches=config.data.n_batches_per_replicate,
            batch_configs=rep_batch_configs,
        )
        
        # Add replicate metadata
        adata.obs['replicate'] = rep_idx
        ground_truth.replicate_id = rep_idx
        
        # Save replicate
        rep_path = os.path.join(output_dir, f"replicate_{rep_idx:02d}.h5ad")
        adata.write_h5ad(rep_path)
        logger.info(f"Saved replicate to {rep_path}")
        
        replicates.append(adata)
        ground_truths.append(ground_truth)
    
    # Save ground truth summary
    gt_summary = {
        'n_replicates': len(ground_truths),
        'interactions': {},
        'de_genes_per_interaction': {}
    }
    
    for gt in ground_truths:
        for name, rule in gt.interactions.items():
            if name not in gt_summary['interactions']:
                gt_summary['interactions'][name] = {
                    'sender_type': rule.sender_type,
                    'receiver_type': rule.receiver_type,
                    'length_scale': rule.length_scale
                }
            if name not in gt_summary['de_genes_per_interaction']:
                gt_summary['de_genes_per_interaction'][name] = []
            gt_summary['de_genes_per_interaction'][name].append(
                len(gt.de_genes.get(name, []))
            )
    
    with open(os.path.join(output_dir, 'ground_truth_summary.json'), 'w') as f:
        json.dump(gt_summary, f, indent=2)
    
    logger.info(f"\nGenerated {len(replicates)} replicates")
    logger.info(f"Data saved to {output_dir}")
    
    return replicates, ground_truths, interactions


# =============================================================================
# Model Training
# =============================================================================

def train_amici_model(
    adata: ad.AnnData,
    config: TrainingConfig,
    use_batch_aware: bool = False,
    use_adversarial: bool = False,
    model_name: str = "amici",
    output_dir: str = "models"
) -> Any:
    """
    Train an AMICI model on the given data.
    
    Parameters
    ----------
    adata : ad.AnnData
        Training data with spatial coordinates and batch labels
    config : TrainingConfig
        Training configuration
    use_batch_aware : bool
        Whether to use batch-aware cross-attention
    use_adversarial : bool
        Whether to use adversarial batch regularization
    model_name : str
        Name for saving the model
    output_dir : str
        Directory to save model checkpoints
        
    Returns
    -------
    Any
        Trained AMICI model
    """
    logger.info(f"\nTraining {model_name}")
    logger.info(f"  Batch-aware: {use_batch_aware}")
    logger.info(f"  Adversarial: {use_adversarial}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Import AMICI
    from amici import AMICI
        
    # Add spatial coordinates to obs if not present
    if 'spatial_x' not in adata.obs.columns and 'spatial' in adata.obsm:
        adata.obs['spatial_x'] = adata.obsm['spatial'][:, 0]
        adata.obs['spatial_y'] = adata.obsm['spatial'][:, 1]
    
    # Split into train/test based on spatial coordinates (or use existing split)
    if 'train_test_split' in adata.obs.columns:
        train_mask = adata.obs['train_test_split'] == 'train'
    else:
        train_mask = ~(
            (adata.obs['spatial_x'] >= 900) & 
            (adata.obs['spatial_x'] <= 1100)
        )
    
    train_adata = adata[train_mask].copy()
    test_adata = adata[~train_mask].copy()
    
    logger.info(f"  Training cells: {train_adata.n_obs}")
    logger.info(f"  Test cells: {test_adata.n_obs}")
    
    # Setup AMICI on training data
    # This registers all necessary fields and computes nearest neighbors
    AMICI.setup_anndata(
        train_adata,
        labels_key="cell_type",
        batch_key="batch",
        coord_obsm_key="spatial",
        n_neighbors=50,
    )
    
    # Create model instance
    # AMICI handles the data loading internally
    model = AMICI(
        train_adata,
        use_adversarial=use_adversarial,
        use_batch_aware=use_batch_aware,
        lambda_adv=config.adversarial_weight if use_adversarial else 0.0,
    )
    
    # Train the model using AMICI's built-in training
    model.train(
        max_epochs=config.n_epochs,
        batch_size=config.batch_size,
        early_stopping=True,
        early_stopping_patience=config.early_stopping_patience,
        check_val_every_n_epoch=1,
        train_size=0.9,  # Use 90% for training, 10% for validation
        # Note: AMICI uses scvi's training infrastructure
    )
    
    # Save model
    model_path = os.path.join(output_dir, f"{model_name}")
    model.save(model_path, overwrite=True)
    logger.info(f"  Model saved to {model_path}")
    
    # Get training history for logging
    try:
        history = model.history
        if 'elbo_train' in history:
            final_train_loss = history['elbo_train'].iloc[-1]
            logger.info(f"  Final training loss: {final_train_loss:.4f}")
        if 'elbo_validation' in history:
            final_val_loss = history['elbo_validation'].iloc[-1]
            logger.info(f"  Final validation loss: {final_val_loss:.4f}")
    except Exception as e:
        logger.warning(f"  Could not retrieve training history: {e}")
    
    logger.info(f"  Training complete.")
    
    return model


def train_models_on_replicate(
    adata: ad.AnnData,
    config: TrainingConfig,
    replicate_id: int,
    output_dir: str
) -> Tuple[Any, Any]:
    """
    Train both baseline and BA-AMICI on a single replicate.
    """
    rep_dir = os.path.join(output_dir, f"replicate_{replicate_id:02d}")
    os.makedirs(rep_dir, exist_ok=True)
    
    # Check if models already exist
    baseline_path = os.path.join(rep_dir, "baseline_amici")
    ba_amici_path = os.path.join(rep_dir, "ba_amici")

    if os.path.exists(baseline_path) and os.path.exists(ba_amici_path):
        logger.info(f"Models already exist for replicate {replicate_id}, loading...")
        try:
            from amici import AMICI
            
            # Setup anndata before loading
            AMICI.setup_anndata(
                adata,
                labels_key="cell_type",
                batch_key="batch",
                coord_obsm_key="spatial",
                n_neighbors=50,
            )
            
            baseline_model = AMICI.load(baseline_path, adata=adata)
            ba_amici_model = AMICI.load(ba_amici_path, adata=adata)
            return baseline_model, ba_amici_model
        except Exception as e:
            logger.warning(f"Could not load models: {e}. Retraining...")
    
    # Train baseline AMICI
    baseline_model = train_amici_model(
        adata=adata,
        config=config,
        use_batch_aware=False,
        use_adversarial=False,
        model_name="baseline_amici",
        output_dir=rep_dir
    )
    
    # Train BA-AMICI
    ba_amici_model = train_amici_model(
        adata=adata,
        config=config,
        use_batch_aware=True,
        use_adversarial=True,
        model_name="ba_amici",
        output_dir=rep_dir
    )
    
    return baseline_model, ba_amici_model


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_models(
    replicates: List[ad.AnnData],
    ground_truths: List[GroundTruth],
    baseline_models: List[Any],
    ba_amici_models: List[Any],
    config: EvaluationConfig,
    interactions_config: Dict[str, Dict],
    output_dir: str
) -> Dict[str, pd.DataFrame]:
    """
    Evaluate both model types across all replicates.
    """
    logger.info("="*60)
    logger.info("EVALUATING MODELS")
    logger.info("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Import AMICI for setup
    try:
        from amici import AMICI
        has_amici = True
    except ImportError:
        has_amici = False
        logger.warning("AMICI not available, predictions will be empty")
    
    # Convert first ground truth to dict format for evaluator
    gt_dict = {
        "interactions": ground_truths[0].interactions,
        "de_genes": ground_truths[0].de_genes,
        "interacting_cells": ground_truths[0].interacting_cells,
    }
    
    # Initialize evaluator with ground truth
    evaluator = InteractionConsistencyEvaluator(ground_truth=gt_dict)
    
    # Extract predictions for all models
    logger.info("\nExtracting predictions...")
    
    replicate_ids = [f"rep_{i:02d}" for i in range(len(replicates))]
    
    for rep_idx, (adata, gt, baseline, ba_amici) in enumerate(
        zip(replicates, ground_truths, baseline_models, ba_amici_models)
    ):
        rep_id = replicate_ids[rep_idx]
        logger.info(f"  Replicate {rep_idx}")
        
        # Setup anndata to compute _nn_idx and _nn_dist
        # These are required for attention pattern extraction
        if has_amici:
            logger.info(f"    Setting up anndata for replicate {rep_idx}...")
            AMICI.setup_anndata(
                adata,
                labels_key="cell_type",
                batch_key="batch",
                coord_obsm_key="spatial",
                n_neighbors=50,
            )
            logger.info(f"    _nn_idx computed: {'_nn_idx' in adata.obsm}")
        
        # Extract baseline predictions
        evaluator.extract_predictions_from_amici(
            model=baseline,
            adata=adata,
            model_name="baseline",
            replicate_id=rep_id,
            interactions_config=interactions_config,
        )
        
        # Extract BA-AMICI predictions
        evaluator.extract_predictions_from_amici(
            model=ba_amici,
            adata=adata,
            model_name="ba_amici",
            replicate_id=rep_id,
            interactions_config=interactions_config,
        )
    
    # Evaluate single-replicate performance
    logger.info("\nEvaluating single-replicate performance...")
    
    baseline_results = []
    ba_amici_results = []
    
    for rep_idx, adata in enumerate(replicates):
        rep_id = replicate_ids[rep_idx]
        
        # Baseline evaluation
        baseline_key = f"baseline_{rep_id}"
        baseline_eval_result = evaluator.evaluate_single_replicate(
            prediction_key=baseline_key,
            adata=adata,
            labels_key="cell_type",
            subtype_key="subtype",
        )
        baseline_results.append({
            'replicate': rep_idx,
            'model': 'baseline',
            'gene_auprc': np.mean(list(baseline_eval_result.gene_auprc.values())) if baseline_eval_result.gene_auprc else 0,
            'sender_auprc': np.mean(list(baseline_eval_result.sender_auprc.values())) if baseline_eval_result.sender_auprc else 0,
            'receiver_auprc': np.mean(list(baseline_eval_result.receiver_auprc.values())) if baseline_eval_result.receiver_auprc else 0,
        })
        
        # BA-AMICI evaluation
        ba_amici_key = f"ba_amici_{rep_id}"
        ba_amici_eval_result = evaluator.evaluate_single_replicate(
            prediction_key=ba_amici_key,
            adata=adata,
            labels_key="cell_type",
            subtype_key="subtype",
        )
        ba_amici_results.append({
            'replicate': rep_idx,
            'model': 'ba_amici',
            'gene_auprc': np.mean(list(ba_amici_eval_result.gene_auprc.values())) if ba_amici_eval_result.gene_auprc else 0,
            'sender_auprc': np.mean(list(ba_amici_eval_result.sender_auprc.values())) if ba_amici_eval_result.sender_auprc else 0,
            'receiver_auprc': np.mean(list(ba_amici_eval_result.receiver_auprc.values())) if ba_amici_eval_result.receiver_auprc else 0,
        })
    
    # Combine single-replicate results
    single_rep_df = pd.DataFrame(baseline_results + ba_amici_results)
    
    # Evaluate cross-replicate consistency
    logger.info("\nEvaluating cross-replicate consistency...")
    
    baseline_consistency = evaluator.evaluate_cross_replicate_consistency(
        model_name="baseline",
        replicate_ids=replicate_ids,
        top_k_interactions=config.top_k_interactions if hasattr(config, 'top_k_interactions') else 100,
    )
    baseline_consistency['model'] = 'baseline'
    
    ba_amici_consistency = evaluator.evaluate_cross_replicate_consistency(
        model_name="ba_amici",
        replicate_ids=replicate_ids,
        top_k_interactions=config.top_k_interactions if hasattr(config, 'top_k_interactions') else 100,
    )
    ba_amici_consistency['model'] = 'ba_amici'
    
    # Combine consistency results
    consistency_df = pd.DataFrame([baseline_consistency, ba_amici_consistency])
    
    # Statistical comparison
    if hasattr(config, 'compute_statistical_tests') and config.compute_statistical_tests:
        logger.info("\nComputing statistical tests...")
        stats_results = compute_statistical_tests(single_rep_df, consistency_df)
    else:
        stats_results = {}
    
    # Generate summary
    summary = generate_results_summary(single_rep_df, consistency_df, stats_results)
    
    # Save results
    single_rep_df.to_csv(os.path.join(output_dir, 'single_replicate_results.csv'), index=False)
    consistency_df.to_csv(os.path.join(output_dir, 'consistency_results.csv'), index=False)
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    if stats_results:
        with open(os.path.join(output_dir, 'statistical_tests.json'), 'w') as f:
            json.dump(stats_results, f, indent=2)
    
    # Generate visualizations
    if hasattr(config, 'generate_visualizations') and config.generate_visualizations:
        logger.info("\nGenerating visualizations...")
        generate_visualizations(single_rep_df, consistency_df, output_dir)
    
    logger.info(f"\nResults saved to {output_dir}")
    
    return {
        'single_replicate': single_rep_df,
        'consistency': consistency_df,
        'summary': summary,
        'statistics': stats_results
    }

def compute_statistical_tests(single_rep_df: pd.DataFrame, consistency_df: pd.DataFrame) -> Dict:
    """
    Compute statistical tests comparing baseline vs BA-AMICI.
    """
    results = {}
    
    # Compare single-replicate metrics
    metrics = ['gene_auprc', 'sender_auprc', 'receiver_auprc']
    
    for metric in metrics:
        if metric not in single_rep_df.columns:
            continue
            
        baseline_values = single_rep_df[single_rep_df['model'] == 'baseline'][metric].values
        ba_amici_values = single_rep_df[single_rep_df['model'] == 'ba_amici'][metric].values
        
        # Check if we have valid data for comparison
        if len(baseline_values) == 0 or len(ba_amici_values) == 0:
            results[f'{metric}_test'] = {'error': 'No data available'}
            continue
        
        # Check if values are all identical (would cause Wilcoxon to fail)
        diff = ba_amici_values - baseline_values
        if np.all(diff == 0) or np.all(np.isnan(diff)):
            results[f'{metric}_test'] = {
                'statistic': None,
                'pvalue': 1.0,
                'note': 'Values are identical or all NaN'
            }
            continue
        
        # Remove NaN pairs
        valid_mask = ~(np.isnan(baseline_values) | np.isnan(ba_amici_values))
        if valid_mask.sum() < 2:
            results[f'{metric}_test'] = {'error': 'Not enough valid pairs'}
            continue
        
        baseline_valid = baseline_values[valid_mask]
        ba_amici_valid = ba_amici_values[valid_mask]
        
        # Check again after removing NaNs
        if np.all(baseline_valid == ba_amici_valid):
            results[f'{metric}_test'] = {
                'statistic': None,
                'pvalue': 1.0,
                'note': 'Values are identical'
            }
            continue
        
        try:
            stat, pval = stats.wilcoxon(ba_amici_valid, baseline_valid)
            results[f'{metric}_test'] = {
                'statistic': float(stat),
                'pvalue': float(pval),
                'ba_amici_mean': float(np.mean(ba_amici_valid)),
                'baseline_mean': float(np.mean(baseline_valid)),
            }
        except Exception as e:
            results[f'{metric}_test'] = {'error': str(e)}
    
    # Compare consistency metrics
    consistency_metrics = ['interaction_jaccard_mean', 'gene_score_correlation_mean', 'attention_correlation_mean']
    
    for metric in consistency_metrics:
        if metric not in consistency_df.columns:
            continue
        
        baseline_val = consistency_df[consistency_df['model'] == 'baseline'][metric].values
        ba_amici_val = consistency_df[consistency_df['model'] == 'ba_amici'][metric].values
        
        if len(baseline_val) > 0 and len(ba_amici_val) > 0:
            baseline_float = float(baseline_val[0]) if not np.isnan(baseline_val[0]) else None
            ba_amici_float = float(ba_amici_val[0]) if not np.isnan(ba_amici_val[0]) else None
            results[f'{metric}_comparison'] = {
                'baseline': baseline_float,
                'ba_amici': ba_amici_float,
            }
    
    return results


def generate_results_summary(
    single_rep_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    stats_results: Dict
) -> Dict[str, Any]:
    """Generate human-readable summary of results."""
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_replicates': single_rep_df['replicate'].nunique(),
        'models_compared': ['baseline', 'ba_amici'],
        'single_replicate_metrics': {},
        'consistency_metrics': {},
        'key_findings': []
    }
    
    # Single replicate summary
    for model in ['baseline', 'ba_amici']:
        model_data = single_rep_df[single_rep_df['model'] == model]
        summary['single_replicate_metrics'][model] = {
            'gene_auprc': {
                'mean': float(model_data['gene_auprc'].mean()),
                'std': float(model_data['gene_auprc'].std())
            },
            'sender_auprc': {
                'mean': float(model_data['sender_auprc'].mean()),
                'std': float(model_data['sender_auprc'].std())
            },
            'receiver_auprc': {
                'mean': float(model_data['receiver_auprc'].mean()),
                'std': float(model_data['receiver_auprc'].std())
            }
        }
    
    # Consistency summary
    for model in ['baseline', 'ba_amici']:
        model_data = consistency_df[consistency_df['model'] == model]
        summary['consistency_metrics'][model] = {
            col: float(model_data[col].values[0]) 
            for col in model_data.columns 
            if col != 'model' and not pd.isna(model_data[col].values[0])
        }
    
    # Key findings
    if stats_results:
        for metric, result in stats_results.items():
            if 'improvement' in result and result['improvement'] is not None:
                if result['improvement'] > 0:
                    summary['key_findings'].append(
                        f"BA-AMICI shows {result['improvement']:.3f} improvement in {metric}"
                    )
                elif result['improvement'] < 0:
                    summary['key_findings'].append(
                        f"Baseline shows {-result['improvement']:.3f} better {metric}"
                    )
            
            if 'p_value' in result and result['p_value'] < 0.05:
                summary['key_findings'].append(
                    f"Significant difference in {metric} (p={result['p_value']:.4f})"
                )
    
    return summary


def generate_visualizations(
    single_rep_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    output_dir: str
):
    """Generate publication-quality visualizations."""
    
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. AUPRC comparison boxplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    metrics = ['gene_auprc', 'sender_auprc', 'receiver_auprc']
    titles = ['Gene Prediction', 'Sender Prediction', 'Receiver Prediction']
    
    for ax, metric, title in zip(axes, metrics, titles):
        if metric in single_rep_df.columns:
            sns.boxplot(
                data=single_rep_df,
                x='model',
                y=metric,
                ax=ax,
                palette=['#1f77b4', '#ff7f0e']
            )
            sns.stripplot(
                data=single_rep_df,
                x='model',
                y=metric,
                ax=ax,
                color='black',
                alpha=0.5,
                size=4
            )
            ax.set_title(title)
            ax.set_xlabel('')
            ax.set_ylabel('AUPRC')
            ax.set_xticklabels(['Baseline', 'BA-AMICI'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'auprc_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, 'auprc_comparison.pdf'), bbox_inches='tight')
    plt.close()
    
    # 2. Consistency comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    
    consistency_metrics = ['interaction_jaccard_mean', 'gene_score_correlation_mean', 'attention_correlation_mean']
    metric_labels = ['Interaction\nJaccard', 'Gene Score\nCorrelation', 'Attention\nCorrelation']
    
    x = np.arange(len(consistency_metrics))
    width = 0.35
    
    baseline_vals = []
    ba_amici_vals = []
    
    for metric in consistency_metrics:
        if metric in consistency_df.columns:
            baseline_vals.append(
                consistency_df[consistency_df['model'] == 'baseline'][metric].values[0]
            )
            ba_amici_vals.append(
                consistency_df[consistency_df['model'] == 'ba_amici'][metric].values[0]
            )
        else:
            baseline_vals.append(0)
            ba_amici_vals.append(0)
    
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='#1f77b4')
    bars2 = ax.bar(x + width/2, ba_amici_vals, width, label='BA-AMICI', color='#ff7f0e')
    
    ax.set_ylabel('Score')
    ax.set_title('Cross-Replicate Consistency')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'consistency_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, 'consistency_comparison.pdf'), bbox_inches='tight')
    plt.close()
    
    # 3. Replicate-wise performance
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for model, marker, color in [('baseline', 'o', '#1f77b4'), ('ba_amici', 's', '#ff7f0e')]:
        model_data = single_rep_df[single_rep_df['model'] == model]
        ax.plot(
            model_data['replicate'],
            model_data['gene_auprc'],
            marker=marker,
            color=color,
            label=f'{model.replace("_", " ").title()} Gene AUPRC',
            linewidth=2,
            markersize=8
        )
    
    ax.set_xlabel('Replicate')
    ax.set_ylabel('Gene AUPRC')
    ax.set_title('Per-Replicate Performance')
    ax.legend()
    ax.set_xticks(range(single_rep_df['replicate'].max() + 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'replicate_performance.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, 'replicate_performance.pdf'), bbox_inches='tight')
    plt.close()
    
    logger.info(f"Figures saved to {fig_dir}")


def load_existing_data(data_dir: str, config: PipelineConfig) -> Tuple[List[ad.AnnData], List[GroundTruth], Dict]:
    """
    Load existing replicates and ground truths if they exist.
    """
    replicates = []
    ground_truths = []
    
    for rep_idx in range(config.data.n_replicates):
        rep_path = os.path.join(data_dir, f"replicate_{rep_idx:02d}.h5ad")
        if os.path.exists(rep_path):
            adata = sc.read_h5ad(rep_path)
            replicates.append(adata)
            
            # Create ground truth from stored info
            gt = GroundTruth()
            gt.replicate_id = rep_idx
            # Note: Full ground truth would need to be loaded from pickle
            ground_truths.append(gt)
        else:
            return None, None, None  # Missing data, need to regenerate
    
    # Rebuild interactions from config
    interactions = {}
    for name, params in config.interactions.interactions.items():
        interactions[name] = {
            "sender": params['sender_type'],
            "receiver": params['receiver_type'],
            "length_scale": params['length_scale'],
        }
    
    return replicates, ground_truths, interactions


def run_pipeline(config: PipelineConfig) -> Dict:
    """
    Run the complete BA-AMICI benchmark pipeline.
    """
    logger.info("="*60)
    logger.info("BA-AMICI BENCHMARK PIPELINE")
    logger.info("="*60)
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Random seed: {config.random_seed}")
    
    # Set random seeds
    np.random.seed(config.random_seed)
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    data_dir = os.path.join(config.output_dir, "data")
    models_dir = os.path.join(config.output_dir, "models")
    eval_dir = os.path.join(config.output_dir, "evaluation")
    
    # Save config
    config.to_yaml(os.path.join(config.output_dir, "config.yaml"))
    
    # =========================================================================
    # STEP 1: Data Generation (or load existing)
    # =========================================================================
    
    # Check if data already exists
    first_rep_path = os.path.join(data_dir, "replicate_00.h5ad")
    last_rep_path = os.path.join(data_dir, f"replicate_{config.data.n_replicates-1:02d}.h5ad")
    
    if os.path.exists(first_rep_path) and os.path.exists(last_rep_path):
        logger.info("="*60)
        logger.info("LOADING EXISTING DATA")
        logger.info("="*60)
        
        replicates = []
        ground_truths = []
        
        for rep_idx in range(config.data.n_replicates):
            rep_path = os.path.join(data_dir, f"replicate_{rep_idx:02d}.h5ad")
            logger.info(f"Loading {rep_path}")
            adata = sc.read_h5ad(rep_path)
            replicates.append(adata)
            
            # Create placeholder ground truth
            gt = GroundTruth()
            gt.replicate_id = rep_idx
            
            # Try to load full ground truth from pickle if exists
            gt_path = os.path.join(data_dir, f"replicate_{rep_idx:02d}.ground_truth.pkl")
            if os.path.exists(gt_path):
                import pickle
                with open(gt_path, 'rb') as f:
                    gt = pickle.load(f)
            
            ground_truths.append(gt)
        
        # Rebuild interactions from config
        interactions = {}
        for name, params in config.interactions.interactions.items():
            interactions[name] = {
                "sender": params['sender_type'],
                "receiver": params['receiver_type'],
                "length_scale": params['length_scale'],
            }
        
        logger.info(f"Loaded {len(replicates)} existing replicates")
    else:
        # Generate new data
        replicates, ground_truths, interactions = generate_benchmark_data(config, data_dir)

    if hasattr(config, 'data_only') and config.data_only:
        logger.info("="*60)
        logger.info("DATA-ONLY MODE: Stopping after data generation")
        logger.info("="*60)
        logger.info(f"Data saved to: {data_dir}")
        return {"replicates": replicates, "ground_truths": ground_truths}
    # =========================================================================
    # STEP 2: Model Training (or load existing)
    # =========================================================================
    logger.info("="*60)
    logger.info("TRAINING MODELS")
    logger.info("="*60)
    
    baseline_models = []
    ba_amici_models = []
    
    for rep_idx, adata in enumerate(replicates):
        logger.info(f"\nTraining on replicate {rep_idx + 1}/{len(replicates)}")
        
        baseline, ba_amici = train_models_on_replicate(
            adata=adata,
            config=config.training,
            replicate_id=rep_idx,
            output_dir=models_dir
        )
        
        baseline_models.append(baseline)
        ba_amici_models.append(ba_amici)
    
    # =========================================================================
    # STEP 3: Evaluation
    # =========================================================================
    logger.info("="*60)
    logger.info("EVALUATING MODELS")
    logger.info("="*60)
    
    results = evaluate_models(
        replicates=replicates,
        ground_truths=ground_truths,
        baseline_models=baseline_models,
        ba_amici_models=ba_amici_models,
        config=config.evaluation,
        interactions_config=interactions,
        output_dir=eval_dir,
    )
    
    # =========================================================================
    # STEP 4: Generate Report
    # =========================================================================
    logger.info("="*60)
    logger.info("GENERATING REPORT")
    logger.info("="*60)
    
    generate_final_report(results, config.output_dir)
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"Results saved to {config.output_dir}")
    
    return results

def generate_final_report(results: Dict, output_dir: str):
    """Generate a final markdown report summarizing the benchmark."""
    
    report_path = os.path.join(output_dir, 'REPORT.md')
    
    summary = results.get('summary', {})
    stats = results.get('statistics', {})
    
    with open(report_path, 'w') as f:
        f.write("# BA-AMICI Benchmark Report\n\n")
        f.write(f"Generated: {summary.get('timestamp', 'N/A')}\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"- Number of replicates: {summary.get('n_replicates', 'N/A')}\n")
        f.write(f"- Models compared: {', '.join(summary.get('models_compared', []))}\n\n")
        
        f.write("## Single-Replicate Performance\n\n")
        f.write("| Model | Gene AUPRC | Sender AUPRC | Receiver AUPRC |\n")
        f.write("|-------|-----------|--------------|----------------|\n")
        
        for model in ['baseline', 'ba_amici']:
            metrics = summary.get('single_replicate_metrics', {}).get(model, {})
            gene = metrics.get('gene_auprc', {})
            sender = metrics.get('sender_auprc', {})
            receiver = metrics.get('receiver_auprc', {})
            
            f.write(f"| {model.replace('_', ' ').title()} | "
                   f"{gene.get('mean', 0):.3f} ± {gene.get('std', 0):.3f} | "
                   f"{sender.get('mean', 0):.3f} ± {sender.get('std', 0):.3f} | "
                   f"{receiver.get('mean', 0):.3f} ± {receiver.get('std', 0):.3f} |\n")
        
        f.write("\n## Cross-Replicate Consistency\n\n")
        f.write("| Model | Interaction Jaccard | Gene Score Correlation |\n")
        f.write("|-------|--------------------|-----------------------|\n")
        
        for model in ['baseline', 'ba_amici']:
            cons = summary.get('consistency_metrics', {}).get(model, {})
            f.write(f"| {model.replace('_', ' ').title()} | "
                   f"{cons.get('interaction_jaccard_mean', 0):.3f} | "
                   f"{cons.get('gene_score_correlation_mean', 0):.3f} |\n")
        
        f.write("\n## Key Findings\n\n")
        for finding in summary.get('key_findings', ['No significant findings']):
            f.write(f"- {finding}\n")
        
        f.write("\n## Statistical Tests\n\n")
        if stats:
            for metric, result in stats.items():
                f.write(f"### {metric}\n\n")
                if 'p_value' in result:
                    f.write(f"- Test: {result.get('test', 'N/A')}\n")
                    f.write(f"- p-value: {result.get('p_value', 'N/A'):.4f}\n")
                    f.write(f"- Baseline: {result.get('baseline_mean', 0):.3f} ± {result.get('baseline_std', 0):.3f}\n")
                    f.write(f"- BA-AMICI: {result.get('ba_amici_mean', 0):.3f} ± {result.get('ba_amici_std', 0):.3f}\n")
                    f.write(f"- Improvement: {result.get('improvement', 0):.3f}\n\n")
                elif 'baseline' in result:
                    f.write(f"- Baseline: {result.get('baseline', 'N/A')}\n")
                    f.write(f"- BA-AMICI: {result.get('ba_amici', 'N/A')}\n")
                    f.write(f"- Improvement: {result.get('improvement', 'N/A')}\n\n")
        else:
            f.write("No statistical tests computed.\n")
        
        f.write("\n## Files\n\n")
        f.write("- `data/`: Generated semi-synthetic datasets\n")
        f.write("- `models/`: Trained model checkpoints\n")
        f.write("- `evaluation/`: Evaluation results and figures\n")
        f.write("- `config.yaml`: Pipeline configuration\n")
    
    logger.info(f"Report saved to {report_path}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface for the benchmark pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="BA-AMICI Benchmark Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python ba_amici_benchmark_pipeline.py
  
  # Run with custom config file
  python ba_amici_benchmark_pipeline.py --config my_config.yaml
  
  # Run with quick test settings
  python ba_amici_benchmark_pipeline.py --quick-test
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results/ba_amici_benchmark',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--n-replicates', '-n',
        type=int,
        default=None,
        help='Override number of replicates'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run with minimal settings for testing'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--data-only',
        action='store_true',
        help='Only generate data, skip training and evaluation'
    )
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()
    
    # Apply overrides
    config.output_dir = args.output_dir
    config.random_seed = args.seed
    
    if args.n_replicates:
        config.data.n_replicates = args.n_replicates
    
    if args.quick_test:
        logger.info("Running in quick-test mode with minimal settings")
        config.data.n_replicates = 2
        config.data.n_cells_per_replicate = 2000
        config.data.n_hvgs = 200
        config.data.use_scvi = False
        config.training.n_epochs = 10
        config.training.early_stopping_patience = 3
    
    # Run pipeline
    results = run_pipeline(config)
    if args.data_only:
        config.data_only = True
 
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    summary = results.get('summary', {})
    for finding in summary.get('key_findings', []):
        print(f"  • {finding}")
    
    print(f"\nFull results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
