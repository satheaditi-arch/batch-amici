import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import anndata as ad
import pytorch_lightning as pl
from amici import AMICI
from amici.callbacks import AttentionPenaltyMonitor

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def main():
    SEED = 42
    pl.seed_everything(SEED)
    
    RUN_ID = "baseline_amici_replicate_00" 
    
    DATA_PATH = os.path.join(str(PROJECT_ROOT), "pbmc_data", "ba_amici_benchmark", "replicate_00.h5ad")
    SAVE_DIR = SCRIPT_DIR / "results"
    MODEL_PATH = SAVE_DIR / RUN_ID

    print(f"Loading data from {DATA_PATH}...")
    adata = ad.read_h5ad(DATA_PATH)
    
    if "spatial" not in adata.obsm:
        if "x_coord" in adata.obs and "y_coord" in adata.obs:
            spatial_coords = np.column_stack((adata.obs["x_coord"].values, adata.obs["y_coord"].values))
            adata.obsm["spatial"] = spatial_coords

    # REMOVED BLINDING - use real batch labels
    
    if "train_test_split" not in adata.obs:
        adata.obs["train_test_split"] = adata.obs["split"] 
        
    adata_train = adata[adata.obs["train_test_split"] == "train"].copy()
    adata_test = adata[adata.obs["train_test_split"] == "test"].copy()

    model_params = {
        "n_heads": 4,
        "n_query_dim": 64,  
        "n_kv_dim": 64,     
        "use_adversarial": False,
        "lambda_adv": 0.0,
        "lambda_pair": 0.001,
        "value_l1_penalty_coef": 0.0, 
    }

    exp_params = {
        "lr": 1e-3,
        "epochs": 50, 
        "batch_size": 128,
        "early_stopping": True,
    }

    print("Setting up AnnData...")
    AMICI.setup_anndata(
        adata_train,
        labels_key="cell_type", 
        batch_key="batch",
        coord_obsm_key="spatial",
        n_neighbors=30
    )

    print("Initializing BASELINE Model (no adversarial training)...")
    model = AMICI(adata_train, **model_params)

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH, exist_ok=True)

    model.train(
        max_epochs=exp_params["epochs"],
        batch_size=exp_params["batch_size"],
        plan_kwargs={"lr": exp_params["lr"]},
        early_stopping=exp_params["early_stopping"],
        check_val_every_n_epoch=1,
        use_wandb=False,
        callbacks=[AttentionPenaltyMonitor(start_val=1e-5, end_val=1e-3, epoch_start=5, epoch_end=40)]
    )

    print(f"Saving baseline to {MODEL_PATH}...")
    model.save(str(MODEL_PATH), overwrite=True)
    print("BASELINE TRAINING COMPLETE.")

if __name__ == "__main__":
    main()