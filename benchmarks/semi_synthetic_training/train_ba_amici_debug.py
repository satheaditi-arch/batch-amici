"""
Diagnostic BA-AMICI Training Script.

Tests different configurations to identify the issue.
Adds extensive logging and validation.
"""

import sys
import os
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import anndata as ad
import pytorch_lightning as pl
from amici import AMICI
from amici.callbacks import AttentionPenaltyMonitor

def setup_directories(base_dir: Path, run_id: str):
    results_dir = base_dir / "results" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def load_and_prepare_data(data_path: Path):
    print(f"\n{'='*60}")
    print(f"Loading data from: {data_path}")
    print(f"{'='*60}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    adata = ad.read_h5ad(data_path)
    
    required_obs = ['cell_type', 'batch', 'split']
    missing = [col for col in required_obs if col not in adata.obs.columns]
    if missing:
        raise ValueError(f"Missing required columns in adata.obs: {missing}")
    
    if 'spatial' not in adata.obsm:
        if 'x_coord' in adata.obs and 'y_coord' in adata.obs:
            coords = np.column_stack([
                adata.obs['x_coord'].values,
                adata.obs['y_coord'].values
            ])
            adata.obsm['spatial'] = coords
        else:
            raise ValueError("No spatial coordinates found")
    
    adata_train = adata[adata.obs['split'] == 'train'].copy()
    adata_test = adata[adata.obs['split'] == 'test'].copy()
    
    print(f"\nDataset Summary:")
    print(f"  Train cells: {adata_train.n_obs}")
    print(f"  Test cells:  {adata_test.n_obs}")
    print(f"  Genes:       {adata_train.n_vars}")
    print(f"  Cell types:  {adata_train.obs['cell_type'].nunique()}")
    print(f"  Batches:     {adata_train.obs['batch'].nunique()}")
    
    return adata_train, adata_test


def run_experiment(config_name, model_config, training_config, data_path):
    """Run a single training experiment with given configuration."""
    
    print("\n" + "="*70)
    print(f"EXPERIMENT: {config_name}")
    print("="*70)
    print(f"Configuration:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")
    
    # Load data
    adata_train, adata_test = load_and_prepare_data(data_path)
    
    # Setup
    AMICI.setup_anndata(
        adata_train,
        labels_key="cell_type",
        batch_key="batch",
        coord_obsm_key="spatial",
        n_neighbors=30
    )
    
    # Initialize model
    model = AMICI(adata_train, **model_config)
    
    # DIAGNOSTIC: Check if adversarial components exist
    if hasattr(model.module, 'discriminator'):
        print("\n✓ Discriminator found in model")
        print(f"  Number of batches: {model.module.n_batches}")
        print(f"  Lambda_adv: {model.module.lambda_adv}")
        print(f"  Lambda_pair: {model.module.lambda_pair}")
    else:
        print("\n✗ No discriminator found in model")
    
    # DIAGNOSTIC: Check max_epochs
    print(f"\nInitial max_epochs: {model.module.max_epochs}")
    print(f"Initial current_epoch: {model.module.current_epoch}")
    
    # Update max_epochs before training
    model.module.max_epochs = training_config["epochs"]
    print(f"Updated max_epochs: {model.module.max_epochs}")
    
    # Create results directory
    results_dir = setup_directories(SCRIPT_DIR, f"debug_{config_name}")
    
    # Training
    print(f"\nStarting training ({training_config['epochs']} epochs)...")
    
    model.train(
        max_epochs=training_config["epochs"],
        batch_size=training_config["batch_size"],
        plan_kwargs={"lr": training_config["lr"]},
        early_stopping=training_config["early_stopping"],
        check_val_every_n_epoch=1,
        use_wandb=False,
        callbacks=[
            AttentionPenaltyMonitor(
                start_val=1e-5,
                end_val=1e-3,
                epoch_start=5,
                epoch_end=min(40, training_config["epochs"] - 10),
                flavor="linear"
            )
        ],
        enable_model_summary=False,
        enable_progress_bar=True,
    )
    
    # DIAGNOSTIC: Check final state
    print(f"\nFinal current_epoch: {model.module.current_epoch}")
    print(f"Final GRL alpha: {model.module._get_grl_alpha()}")
    
    # Save
    model.save(str(results_dir), overwrite=True)
    print(f"✓ Saved to {results_dir}")
    
    # Quick evaluation
    AMICI.setup_anndata(
        adata_test,
        labels_key="cell_type",
        batch_key="batch",
        coord_obsm_key="spatial",
        n_neighbors=30
    )
    
    test_metrics = model.get_reconstruction_error(adata_test, batch_size=128)
    print(f"\nTest Reconstruction Loss: {test_metrics['reconstruction_loss']:.4f}")
    
    return results_dir, test_metrics


def main():
    # Seed everything
    SEED = 42
    pl.seed_everything(SEED, workers=True)
    
    # Paths
    DATA_DIR = PROJECT_ROOT / "pbmc_data" / "ba_amici_benchmark"
    REPLICATE_ID = "replicate_00"
    DATA_PATH = DATA_DIR / f"{REPLICATE_ID}.h5ad"
    
    # Base training config
    base_training_config = {
        "lr": 1e-3,
        "epochs": 50,  # Shorter for debugging
        "batch_size": 128,
        "early_stopping": True,
    }
    
    # ====================================================================
    # EXPERIMENT 1: No Adversarial (Baseline Reproduction)
    # ====================================================================
    exp1_config = {
        "n_heads": 4,
        "n_query_dim": 64,  
        "n_kv_dim": 64,
        "use_adversarial": False,
        "lambda_adv": 0.0,
        "lambda_pair": 0.0,
        "value_l1_penalty_coef": 0.0,
    }
    
    print("\n" + "#"*70)
    print("# EXPERIMENT 1: Baseline (No Adversarial)")
    print("#"*70)
    
    exp1_dir, exp1_metrics = run_experiment(
        "exp1_no_adversarial",
        exp1_config,
        base_training_config,
        DATA_PATH
    )
    
    # ====================================================================
    # EXPERIMENT 2: Low Lambda Adversarial
    # ====================================================================
    exp2_config = {
        "n_heads": 4,
        "n_query_dim": 64,  
        "n_kv_dim": 64,
        "use_adversarial": True,
        "lambda_adv": 0.1,  # Much lower
        "lambda_pair": 0.001,
        "value_l1_penalty_coef": 0.0,
    }
    
    print("\n" + "#"*70)
    print("# EXPERIMENT 2: Low Lambda Adversarial (0.1)")
    print("#"*70)
    
    exp2_dir, exp2_metrics = run_experiment(
        "exp2_low_lambda",
        exp2_config,
        base_training_config,
        DATA_PATH
    )
    
    # ====================================================================
    # EXPERIMENT 3: Medium Lambda Adversarial
    # ====================================================================
    exp3_config = {
        "n_heads": 4,
        "n_query_dim": 64,  
        "n_kv_dim": 64,
        "use_adversarial": True,
        "lambda_adv": 0.5,
        "lambda_pair": 0.001,
        "value_l1_penalty_coef": 0.0,
    }
    
    print("\n" + "#"*70)
    print("# EXPERIMENT 3: Medium Lambda Adversarial (0.5)")
    print("#"*70)
    
    exp3_dir, exp3_metrics = run_experiment(
        "exp3_medium_lambda",
        exp3_config,
        base_training_config,
        DATA_PATH
    )
    
    # ====================================================================
    # EXPERIMENT 4: Original High Lambda (Your Current Config)
    # ====================================================================
    exp4_config = {
        "n_heads": 4,
        "n_query_dim": 64,  
        "n_kv_dim": 64,
        "use_adversarial": True,
        "lambda_adv": 1.0,
        "lambda_pair": 0.001,
        "value_l1_penalty_coef": 0.0,
    }
    
    print("\n" + "#"*70)
    print("# EXPERIMENT 4: High Lambda Adversarial (1.0 - Original)")
    print("#"*70)
    
    exp4_dir, exp4_metrics = run_experiment(
        "exp4_high_lambda",
        exp4_config,
        base_training_config,
        DATA_PATH
    )
    
    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    results = [
        ("Exp1: No Adversarial", exp1_metrics['reconstruction_loss']),
        ("Exp2: Lambda=0.1", exp2_metrics['reconstruction_loss']),
        ("Exp3: Lambda=0.5", exp3_metrics['reconstruction_loss']),
        ("Exp4: Lambda=1.0", exp4_metrics['reconstruction_loss']),
    ]
    
    print("\nReconstruction Loss Comparison:")
    for name, loss in results:
        print(f"  {name}: {loss:.4f}")
    
    print("\nNext Steps:")
    print("1. Run validation on each experiment:")
    print(f"   Edit validate_batch_correction.py to point to debug_exp2_low_lambda, etc.")
    print("2. Compare metrics to see which lambda works best")
    print("3. Check training logs for adversarial loss values")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()