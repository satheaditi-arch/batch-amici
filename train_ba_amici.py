import sys
import os
import numpy as np

# --- Connect to the 'src' folder ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
sys.path.append(src_path)
# -----------------------------------

import anndata as ad
import pytorch_lightning as pl
from amici import AMICI
from amici.callbacks import AttentionPenaltyMonitor

def main():
    # 1. Configuration
    SEED = 42
    pl.seed_everything(SEED)
    
    DATA_PATH = os.path.join(current_dir, "ba_amici_benchmark", "replicate_00.h5ad")
    RUN_ID = "ba_amici_test_run"
    
    # Force absolute path for results
    SAVE_DIR = os.path.join(current_dir, "results")
    MODEL_PATH = os.path.join(SAVE_DIR, RUN_ID)

    # 2. Load Data
    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: File not found at {DATA_PATH}")
        return

    adata = ad.read_h5ad(DATA_PATH)
    
    # Fix spatial matrix
    if "spatial" not in adata.obsm:
        if "x" in adata.obs and "y" in adata.obs:
            spatial_coords = np.column_stack((adata.obs["x"].values, adata.obs["y"].values))
            adata.obsm["spatial"] = spatial_coords

    # Ensure split
    if "train_test_split" not in adata.obs:
        adata.obs["train_test_split"] = adata.obs["split"] 
        
    adata_train = adata[adata.obs["train_test_split"] == "train"].copy()
    adata_test = adata[adata.obs["train_test_split"] == "test"].copy()

    # 3. Define Parameters
    model_params = {
        "n_heads": 4,
        "n_query_dim": 64,  
        "n_kv_dim": 64,     
        "use_adversarial": True,
        "lambda_adv": 1.0,        
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
    print("Initializing BA-AMICI Model...")
    model = AMICI(adata_train, **model_params)

    print(f"Starting training (Target: {MODEL_PATH})...")
    
    # --- FORCE DIRECTORY CREATION ---
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH, exist_ok=True)
    # --------------------------------

    model.train(
        max_epochs=exp_params["epochs"],
        batch_size=exp_params["batch_size"],
        plan_kwargs={"lr": exp_params["lr"]},
        early_stopping=exp_params["early_stopping"],
        check_val_every_n_epoch=1,
        use_wandb=False,
        callbacks=[
            AttentionPenaltyMonitor(
                start_val=1e-5,
                end_val=1e-3,
                epoch_start=5,
                epoch_end=40,
                flavor="linear"
            )
        ]
    )

    # 6. Save
    print(f"Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH, overwrite=True)
    
    # 7. Verify Save
    if os.path.exists(os.path.join(MODEL_PATH, "model.pt")):
        print("✅ SUCCESS! Model file found on disk.")
    else:
        print("❌ ERROR! Model file NOT found after saving.")

    # 8. Evaluate (The step I added back!)
    print("Evaluating on Test Set...")
    AMICI.setup_anndata(
        adata_test, 
        labels_key="cell_type", 
        batch_key="batch",
        coord_obsm_key="spatial",
        n_neighbors=30
    )
    
    # Calculate reconstruction loss
    metrics = model.get_reconstruction_error(adata_test, batch_size=128)
    print(f"Test Reconstruction Loss: {metrics['reconstruction_loss']}")

if __name__ == "__main__":
    main()