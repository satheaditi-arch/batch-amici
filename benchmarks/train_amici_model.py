"""
Training script for Baseline AMICI (NO batch correction).

This serves as a control to measure:
1. How much batch effects confound spatial interaction inference
2. The benefit of adversarial batch correction (BA-AMICI vs Baseline)

NOTE: We blind the model by setting all batch labels to 'batch_0'.
This simulates the original AMICI which has no batch-awareness.
"""

import sys
import os
import numpy as np
from pathlib import Path

# --- Path Setup ---
SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_PATH = SCRIPT_DIR / "src"
sys.path.insert(0, str(SRC_PATH))
# ------------------

import anndata as ad
import pytorch_lightning as pl
from amici import AMICI
from amici.callbacks import AttentionPenaltyMonitor

def setup_directories(base_dir: Path, run_id: str):
    """Create directory structure for experiment."""
    results_dir = base_dir / "results" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def load_and_blind_data(data_path: Path):
    """
    Load benchmark data and BLIND batch information.
    
    This simulates a model that has no batch correction capability.
    All cells are assigned to a dummy 'batch_0' label.
    
    Returns
    -------
    adata_train, adata_test : AnnData
        Split datasets with blinded batch labels
    """
    print(f"\n{'='*60}")
    print(f"Loading data from: {data_path}")
    print(f"{'='*60}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    adata = ad.read_h5ad(data_path)
    
    # Ensure spatial coordinates exist
    if 'spatial' not in adata.obsm:
        if 'x_coord' in adata.obs and 'y_coord' in adata.obs:
            coords = np.column_stack([
                adata.obs['x_coord'].values,
                adata.obs['y_coord'].values
            ])
            adata.obsm['spatial'] = coords
            print("✓ Created spatial coordinates from x_coord/y_coord")
        else:
            raise ValueError("No spatial coordinates found")
    
    # CRITICAL: Blind the batch information
    print("\n" + "!"*60)
    print("BLINDING BATCH LABELS")
    print("!"*60)
    print("Setting all batch labels to 'batch_0'")
    print("This removes batch information from the model input.")
    print("The model will learn interactions WITHOUT batch correction.")
    print("!"*60 + "\n")
    
    original_batches = adata.obs['batch'].value_counts()
    print("Original batch distribution:")
    for batch, count in original_batches.items():
        print(f"  {batch}: {count} cells")
    
    # Overwrite with dummy batch
    adata.obs['batch'] = 'batch_0'
    adata.obs['batch'] = adata.obs['batch'].astype('category')
    
    # Split data
    adata_train = adata[adata.obs['split'] == 'train'].copy()
    adata_test = adata[adata.obs['split'] == 'test'].copy()
    
    print(f"\nDataset Summary:")
    print(f"  Train cells: {adata_train.n_obs}")
    print(f"  Test cells:  {adata_test.n_obs}")
    print(f"  Genes:       {adata_train.n_vars}")
    print(f"  Cell types:  {adata_train.obs['cell_type'].nunique()}")
    print(f"  Batches:     {adata_train.obs['batch'].nunique()} (blinded)")
    
    return adata_train, adata_test

def main():
    # ==================== CONFIGURATION ====================
    SEED = 42
    pl.seed_everything(SEED, workers=True)
    
    # Paths - using pbmc_data/ba_amici_benchmark structure
    DATA_DIR = SCRIPT_DIR / "pbmc_data" / "ba_amici_benchmark"
    REPLICATE_ID = "replicate_00"
    DATA_PATH = DATA_DIR / f"{REPLICATE_ID}.h5ad"
    
    RUN_ID = f"baseline_amici_{REPLICATE_ID}"
    
    # Model hyperparameters (BASELINE CONFIG)
    model_config = {
        # Architecture (same as BA-AMICI for fair comparison)
        "n_heads": 4,
        "n_query_dim": 64,  
        "n_kv_dim": 64,
        
        # NO BATCH CORRECTION
        "use_adversarial": False,  # Disabled
        "lambda_adv": 0.0,         # Zero weight
        
        # Regularization
        "value_l1_penalty_coef": 0.0,
    }
    
    training_config = {
        "lr": 1e-3,
        "epochs": 100,
        "batch_size": 128,
        "early_stopping": True,
        "patience": 10,
    }
    
    penalty_schedule = {
        "start_val": 1e-5,
        "end_val": 1e-3,
        "epoch_start": 5,
        "epoch_end": 80,
        "flavor": "linear"
    }
    
    # ==================== SETUP ====================
    results_dir = setup_directories(SCRIPT_DIR, RUN_ID)
    print(f"\n{'='*60}")
    print(f"BASELINE AMICI Training (No Batch Correction)")
    print(f"Results will be saved to: {results_dir}")
    print(f"{'='*60}")
    
    # ==================== DATA LOADING ====================
    adata_train, adata_test = load_and_blind_data(DATA_PATH)
    
    # ==================== MODEL SETUP ====================
    print(f"\n{'='*60}")
    print("Setting up AMICI for training data...")
    print(f"{'='*60}")
    
    AMICI.setup_anndata(
        adata_train,
        labels_key="cell_type",
        batch_key="batch",  # Will be 'batch_0' for all cells
        coord_obsm_key="spatial",
        n_neighbors=30
    )
    
    print("\nInitializing BASELINE model...")
    print(f"Configuration:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")
    
    model = AMICI(adata_train, **model_config)
    
    # ==================== TRAINING ====================
    print(f"\n{'='*60}")
    print(f"Starting Training (Epochs: {training_config['epochs']})")
    print(f"{'='*60}\n")
    
    model.train(
        max_epochs=training_config["epochs"],
        batch_size=training_config["batch_size"],
        plan_kwargs={"lr": training_config["lr"]},
        early_stopping=training_config["early_stopping"],
        check_val_every_n_epoch=1,
        use_wandb=False,
        callbacks=[
            AttentionPenaltyMonitor(**penalty_schedule)
        ],
        enable_model_summary=True,
        enable_progress_bar=True,
    )
    
    # ==================== SAVE MODEL ====================
    print(f"\n{'='*60}")
    print(f"Saving model to: {results_dir}")
    print(f"{'='*60}")
    
    model.save(str(results_dir), overwrite=True)
    
    # Verify save
    model_file = results_dir / "model.pt"
    if model_file.exists():
        print(f"✓ Model saved successfully ({model_file.stat().st_size / 1024:.1f} KB)")
    else:
        print("✗ WARNING: Model file not found after save!")
    
    # ==================== EVALUATION ====================
    print(f"\n{'='*60}")
    print("Evaluating on Test Set")
    print(f"{'='*60}")
    
    # Setup test data (also blinded)
    AMICI.setup_anndata(
        adata_test,
        labels_key="cell_type",
        batch_key="batch",
        coord_obsm_key="spatial",
        n_neighbors=30
    )
    
    # Compute metrics
    test_metrics = model.get_reconstruction_error(
        adata_test,
        batch_size=training_config["batch_size"]
    )
    
    print(f"\nTest Metrics:")
    print(f"  Reconstruction Loss: {test_metrics['reconstruction_loss']:.4f}")
    
    # Get embeddings
    print("\nExtracting embeddings (no batch correction applied)...")
    embeddings = model.get_latent_representation(adata_test)
    
    # Save embeddings
    embedding_path = results_dir / "test_embeddings.npy"
    np.save(embedding_path, embeddings)
    print(f"✓ Embeddings saved to {embedding_path}")
    
    # ==================== SUMMARY ====================
    print(f"\n{'='*60}")
    print("BASELINE Training Complete!")
    print(f"{'='*60}")
    print(f"Model:      {results_dir / 'model.pt'}")
    print(f"Embeddings: {embedding_path}")
    print(f"\nThis baseline has NO batch correction.")
    print(f"Compare with BA-AMICI to measure improvement.")
    print(f"\nNext steps:")
    print(f"  1. Ensure you've run train_ba_amici.py")
    print(f"  2. Use benchmarks/compare_models.py to quantify differences")

if __name__ == "__main__":
    main()