import sys
import os
import numpy as np
import pandas as pd

# --- Connect to src ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
sys.path.append(src_path)
# --------------------

import anndata as ad
import pytorch_lightning as pl
from amici import AMICI
from amici.callbacks import AttentionPenaltyMonitor

def main():
    # 1. Config - NOTE THE NEW ID
    SEED = 42
    pl.seed_everything(SEED)
    RUN_ID = "baseline_amici_run"  # <--- Saving to a different folder
    
    DATA_PATH = os.path.join(current_dir, "ba_amici_benchmark", "replicate_00.h5ad")
    SAVE_DIR = os.path.join(current_dir, "results")
    MODEL_PATH = os.path.join(SAVE_DIR, RUN_ID)

    # 2. Load Data
    print(f"Loading data from {DATA_PATH}...")
    adata = ad.read_h5ad(DATA_PATH)
    
    # Fix spatial
    if "spatial" not in adata.obsm:
        if "x" in adata.obs and "y" in adata.obs:
            spatial_coords = np.column_stack((adata.obs["x"].values, adata.obs["y"].values))
            adata.obsm["spatial"] = spatial_coords

    # --- THE BLINDING STEP (Simulating Original AMICI) ---
    print("BLINDING MODEL: Overwriting batch labels to 0...")
    # We force every cell to belong to 'batch_0'. 
    # This removes all batch information from the input.
    adata.obs['batch'] = 'batch_0'
    adata.obs['batch'] = adata.obs['batch'].astype('category')
    # -----------------------------------------------------

    if "train_test_split" not in adata.obs:
        adata.obs["train_test_split"] = adata.obs["split"] 
        
    adata_train = adata[adata.obs["train_test_split"] == "train"].copy()
    adata_test = adata[adata.obs["train_test_split"] == "test"].copy()

    # 3. Define Parameters (BASELINE CONFIG)
    model_params = {
        "n_heads": 4,
        "n_query_dim": 64,  
        "n_kv_dim": 64,     
        "use_adversarial": False, # <--- DISABLED (No Aim 2)
        "lambda_adv": 0.0,        # <--- Zero weight
        "value_l1_penalty_coef": 0.0, 
    }

    exp_params = {
        "lr": 1e-3,
        "epochs": 50, 
        "batch_size": 128,
        "early_stopping": True,
    }

    # 4. Setup
    print("Setting up AnnData...")
    AMICI.setup_anndata(
        adata_train,
        labels_key="cell_type", 
        batch_key="batch",       
        coord_obsm_key="spatial",
        n_neighbors=30
    )

    # 5. Initialize & Train
    print("Initializing BASELINE Model...")
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

    # 6. Save
    print(f"Saving baseline to {MODEL_PATH}...")
    model.save(MODEL_PATH, overwrite=True)
    print("BASELINE TRAINING COMPLETE.")

if __name__ == "__main__":
    main()