"""
BA-AMICI Training Script.

Includes:
- Proper max_epochs setting
- Diagnostic logging
- MaxEpochsSetter callback
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
            print(f"ADVERSARIAL TRAINING DIAGNOSTICS")
            print(f"{'='*60}")
            print(f"✓ Set max_epochs = {trainer.max_epochs}")
            
            if hasattr(pl_module.module, 'use_adversarial'):
                print(f"✓ Adversarial enabled: {pl_module.module.use_adversarial}")
                if pl_module.module.use_adversarial:
                    print(f"  lambda_adv: {pl_module.module.lambda_adv}")
                    print(f"  lambda_pair: {pl_module.module.lambda_pair}")
                    print(f"  Initial GRL alpha: {pl_module.module._get_grl_alpha():.3f}")
                    
                    # Check discriminator exists
                    if hasattr(pl_module.module, 'discriminator'):
                        print(f"✓ Discriminator initialized")
                    else:
                        print(f"✗ WARNING: No discriminator found!")
            print(f"{'='*60}\n")
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Update epoch counter in module."""
        if hasattr(pl_module.module, 'on_train_epoch_end'):
            pl_module.module.on_train_epoch_end()
            
            # Log every 10 epochs
            if trainer.current_epoch % 10 == 0:
                if hasattr(pl_module.module, 'use_adversarial') and pl_module.module.use_adversarial:
                    alpha = pl_module.module._get_grl_alpha()
                    print(f"Epoch {trainer.current_epoch}: GRL alpha = {alpha:.3f}")


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


def main():
    SEED = 42
    pl.seed_everything(SEED, workers=True)
    
    DATA_DIR = PROJECT_ROOT / "pbmc_data" / "ba_amici_benchmark"
    REPLICATE_ID = "replicate_00"
    DATA_PATH = DATA_DIR / f"{REPLICATE_ID}.h5ad"
    
    RUN_ID = f"ba_amici_{REPLICATE_ID}_v2"  # v2 to avoid overwriting
    
    # UPDATED CONFIG with lower lambda
    model_config = {
        "n_heads": 4,
        "n_query_dim": 64,  
        "n_kv_dim": 64,
        "use_adversarial": True,
        "lambda_adv": 0.05,  # REDUCED from 1.0
        "lambda_pair": 0.001,  # Added explicit gamma regularization
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
    
    results_dir = setup_directories(SCRIPT_DIR, RUN_ID)
    print(f"\n{'='*60}")
    print(f"BA-AMICI Training (FIXED VERSION)")
    print(f"Results will be saved to: {results_dir}")
    print(f"{'='*60}")
    
    adata_train, adata_test = load_and_prepare_data(DATA_PATH)
    
    print(f"\n{'='*60}")
    print("Setting up AMICI for training data...")
    print(f"{'='*60}")
    
    AMICI.setup_anndata(
        adata_train,
        labels_key="cell_type",
        batch_key="batch",
        coord_obsm_key="spatial",
        n_neighbors=30
    )
    
    print("\nInitializing BA-AMICI model...")
    print(f"Configuration:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")
    
    model = AMICI(adata_train, **model_config)
    
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
            MaxEpochsSetter(),  # ADDED: Ensures max_epochs is set
            AttentionPenaltyMonitor(**penalty_schedule)
        ],
        enable_model_summary=True,
        enable_progress_bar=True,
    )
    
    print(f"\n{'='*60}")
    print(f"Saving model to: {results_dir}")
    print(f"{'='*60}")
    
    model.save(str(results_dir), overwrite=True)
    
    model_file = results_dir / "model.pt"
    if model_file.exists():
        print(f"✓ Model saved successfully ({model_file.stat().st_size / 1024:.1f} KB)")
    else:
        print("✗ WARNING: Model file not found after save!")
    
    print(f"\n{'='*60}")
    print("Evaluating on Test Set")
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

    print(f"\nTest Metrics:")
    print(f"  Reconstruction Loss: {test_metrics['reconstruction_loss']:.4f}")

    print("\nExtracting batch-corrected embeddings...")
    try:
        corrected_embeddings = model.get_nn_embed(
            adata_test, 
            batch_size=training_config["batch_size"]
        )
        
        embedding_path = results_dir / "test_embeddings.npy"
        np.save(embedding_path, corrected_embeddings)
        print(f"✓ Embeddings saved to {embedding_path}")
        print(f"  Shape: {corrected_embeddings.shape}")
        
    except Exception as e:
        print(f"Could not extract embeddings: {e}")
        corrected_embeddings = None
        
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Model:      {results_dir / 'model.pt'}")

if __name__ == "__main__":
    main()