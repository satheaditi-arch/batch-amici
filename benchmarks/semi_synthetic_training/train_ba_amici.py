"""
BA-AMICI Training Script - FIXED VERSION

Key Settings:
- use_batch_aware=True -> Uses BatchAwareCrossAttention
- use_adversarial=True -> Adds adversarial batch discrimination loss

This ensures proper batch-aware training.
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
from pytorch_lightning.callbacks import Callback
from amici import AMICI
from amici.callbacks import AttentionPenaltyMonitor


class MaxEpochsSetter(Callback):
    """Ensures max_epochs is set in module for GRL scheduling."""
    
    def on_train_start(self, trainer, pl_module):
        if hasattr(pl_module.module, 'max_epochs'):
            pl_module.module.max_epochs = trainer.max_epochs
            print(f"\n{'='*60}")
            print(f"BA-AMICI TRAINING DIAGNOSTICS")
            print(f"{'='*60}")
            print(f"✓ Set max_epochs = {trainer.max_epochs}")
            
            if hasattr(pl_module.module, 'use_batch_aware'):
                print(f"✓ Batch-aware attention: {pl_module.module.use_batch_aware}")
            
            if hasattr(pl_module.module, 'use_adversarial'):
                print(f"✓ Adversarial training: {pl_module.module.use_adversarial}")
                if pl_module.module.use_adversarial:
                    print(f"  lambda_adv: {pl_module.module.lambda_adv}")
                    print(f"  Initial GRL alpha: {pl_module.module._get_grl_alpha():.3f}")
            print(f"{'='*60}\n")
    
    def on_train_epoch_end(self, trainer, pl_module):
        if hasattr(pl_module.module, 'on_train_epoch_end'):
            pl_module.module.on_train_epoch_end()
            
            # Log every 20 epochs
            if trainer.current_epoch % 20 == 0 and trainer.current_epoch > 0:
                if hasattr(pl_module.module, 'use_adversarial') and pl_module.module.use_adversarial:
                    alpha = pl_module.module._get_grl_alpha()
                    print(f"Epoch {trainer.current_epoch}: GRL alpha = {alpha:.3f}")


def main():
    SEED = 42
    pl.seed_everything(SEED, workers=True)
    
    # ==================== CONFIGURATION ====================
    RUN_ID = "ba_amici_replicate_00"
    
    DATA_PATH = PROJECT_ROOT / "pbmc_data" / "ba_amici_benchmark_v2" / "replicate_00.h5ad"
    SAVE_DIR = SCRIPT_DIR / "results"
    MODEL_PATH = SAVE_DIR / RUN_ID
    
    # BA-AMICI CONFIG: Full batch awareness
    model_config = {
        "n_heads": 4,
        "n_query_dim": 64,
        "n_kv_dim": 64,
        # KEY SETTINGS FOR BA-AMICI:
        "use_batch_aware": True,   # Uses BatchAwareCrossAttention
        "use_adversarial": True,   # Adversarial batch discrimination
        "lambda_adv": 0.1,         # Adversarial loss weight
        "lambda_pair": 0.001,      # Batch-pair bias regularization
        "value_l1_penalty_coef": 0.0,
    }
    
    training_config = {
        "lr": 1e-3,
        "epochs": 100,
        "batch_size": 128,
        "early_stopping": True,
        "patience": 15,  # More patience for adversarial training
    }
    
    penalty_schedule = {
        "start_val": 1e-5,
        "end_val": 1e-3,
        "epoch_start": 5,
        "epoch_end": 80,
        "flavor": "linear"
    }
    
    # ==================== LOAD DATA ====================
    print(f"\n{'='*60}")
    print("BA-AMICI Training (BATCH-AWARE + ADVERSARIAL)")
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
    
    print("\nInitializing BA-AMICI model (BatchAwareCrossAttention + Adversarial)...")
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
            MaxEpochsSetter(),
            AttentionPenaltyMonitor(**penalty_schedule)
        ],
        enable_model_summary=True,
        enable_progress_bar=True,
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
    
    # Extract and save embeddings
    print("\nExtracting embeddings for evaluation...")
    try:
        embeddings = model.get_predictions(adata_test, get_residuals=True, batch_size=128)
        embedding_path = MODEL_PATH / "test_embeddings.npy"
        np.save(embedding_path, embeddings)
        print(f"✓ Embeddings saved: {embedding_path}")
    except Exception as e:
        print(f"Could not save embeddings: {e}")
    
    print(f"\n{'='*60}")
    print("BA-AMICI TRAINING COMPLETE")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()