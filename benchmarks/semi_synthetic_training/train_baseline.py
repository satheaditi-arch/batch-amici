"""
BASELINE AMICI Training Script - FIXED VERSION

Key Difference from BA-AMICI:
- use_batch_aware=False -> Uses StandardCrossAttention (NO batch info)
- use_adversarial=False -> No adversarial loss

This ensures the baseline is truly batch-UNAWARE.
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


def main():
    SEED = 42
    pl.seed_everything(SEED)
    
    # ==================== CONFIGURATION ====================
    RUN_ID = "baseline_amici_replicate_00"
    
    DATA_PATH = PROJECT_ROOT / "pbmc_data" / "ba_amici_benchmark_v2" / "replicate_00.h5ad"
    SAVE_DIR = SCRIPT_DIR / "results"
    MODEL_PATH = SAVE_DIR / RUN_ID
    
    # BASELINE CONFIG: NO batch awareness
    model_config = {
        "n_heads": 4,
        "n_query_dim": 64,
        "n_kv_dim": 64,
        # KEY SETTINGS FOR BASELINE:
        "use_batch_aware": False,  # NEW: Uses StandardCrossAttention
        "use_adversarial": False,  # No adversarial loss
        "lambda_adv": 0.0,
        "lambda_pair": 0.0,
        "value_l1_penalty_coef": 0.0,
    }
    
    training_config = {
        "lr": 1e-3,
        "epochs": 100,
        "batch_size": 128,
        "early_stopping": True,
        "patience": 10,
    }
    
    # ==================== LOAD DATA ====================
    print(f"\n{'='*60}")
    print("BASELINE AMICI Training (NO BATCH AWARENESS)")
    print(f"{'='*60}")
    
    print(f"\nLoading data from {DATA_PATH}...")
    adata = ad.read_h5ad(DATA_PATH)
    
    # Fix spatial coordinates
    if "spatial" not in adata.obsm:
        if "x_coord" in adata.obs and "y_coord" in adata.obs:
            adata.obsm["spatial"] = np.column_stack([
                adata.obs["x_coord"].values,
                adata.obs["y_coord"].values
            ])
    
    # Split data
    adata_train = adata[adata.obs["split"] == "train"].copy()
    adata_test = adata[adata.obs["split"] == "test"].copy()
    
    print(f"Train: {adata_train.n_obs} cells")
    print(f"Test:  {adata_test.n_obs} cells")
    print(f"Genes: {adata_train.n_vars}")
    print(f"Cell types: {adata_train.obs['cell_type'].nunique()}")
    print(f"Batches: {adata_train.obs['batch'].nunique()}")
    
    # ==================== SETUP & TRAIN ====================
    print("\nSetting up AnnData...")
    AMICI.setup_anndata(
        adata_train,
        labels_key="cell_type",
        batch_key="batch",
        coord_obsm_key="spatial",
        n_neighbors=30
    )
    
    print("\nInitializing BASELINE model (StandardCrossAttention)...")
    print("Configuration:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")
    
    model = AMICI(adata_train, **model_config)
    
    # Create output directory
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training for {training_config['epochs']} epochs...")
    print(f"{'='*60}\n")
    
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
                epoch_end=80
            )
        ]
    )
    
    # ==================== SAVE ====================
    print(f"\nSaving model to {MODEL_PATH}...")
    model.save(str(MODEL_PATH), overwrite=True)
    
    # ==================== EVALUATE ====================
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    print(f"{'='*60}")
    
    AMICI.setup_anndata(
        adata_test,
        labels_key="cell_type",
        batch_key="batch",
        coord_obsm_key="spatial",
        n_neighbors=30
    )
    
    test_metrics = model.get_reconstruction_error(
        adata_test,
        batch_size=training_config["batch_size"]
    )
    print(f"Test reconstruction loss: {test_metrics['reconstruction_loss']:.4f}")
    
    print(f"\n{'='*60}")
    print("BASELINE TRAINING COMPLETE")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()