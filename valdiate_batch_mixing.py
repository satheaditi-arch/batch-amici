import sys
import os
import torch
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np

# --- Connect to src ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
sys.path.append(src_path)
# --------------------

from amici import AMICI

def main():
    # 1. Config
    RUN_ID = "ba_amici_test_run"
    # Ensure we point to the absolute path where the model is saved
    MODEL_PATH = os.path.join(current_dir, "results", RUN_ID)
    DATA_PATH = os.path.join(current_dir, "ba_amici_benchmark", "replicate_00.h5ad")
    
    # 2. Load Data
    print(f"Loading data from {DATA_PATH}...")
    adata = sc.read_h5ad(DATA_PATH)
    
    # Fix spatial if needed (glue x/y columns)
    if "spatial" not in adata.obsm:
        if "x" in adata.obs and "y" in adata.obs:
            adata.obsm["spatial"] = np.column_stack((adata.obs["x"].values, adata.obs["y"].values))

    # 3. Load Trained Model
    print(f"Loading model from {MODEL_PATH}...")
    # We must match the training setup exactly
    AMICI.setup_anndata(
        adata, 
        labels_key="cell_type", 
        batch_key="batch", 
        coord_obsm_key="spatial", 
        n_neighbors=30
    )
    model = AMICI.load(MODEL_PATH, adata=adata)
    
    # 4. Extract Biological Residuals (The Fix!)
    print("Extracting biological residuals (Delta)...")
    # We fetch the 'Residuals' which represent the learned biological shift
    # Aim 2 forces these residuals to be batch-invariant
    residuals = model.get_predictions(adata, batch_size=128, get_residuals=True)
    
    # Store in adata for plotting
    adata.obsm["X_ba_amici_residuals"] = residuals

    # 5. Visualize with UMAP
    print("Computing neighbors on residuals...")
    sc.pp.neighbors(adata, use_rep="X_ba_amici_residuals")
    
    print("Computing UMAP...")
    sc.tl.umap(adata)

    # 6. Plot
    print("Plotting results...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot A: Colored by Cell Type 
    # Goal: Distinct Clusters (Biology is preserved)
    sc.pl.umap(adata, color="cell_type", ax=axes[0], show=False, title="Cell Type (Biology)")
    
    # Plot B: Colored by Batch
    # Goal: MIXED (Batch effect is removed)
    # If Aim 2 (Adversarial) worked, this should look like a blended smoothie.
    sc.pl.umap(adata, color="batch", ax=axes[1], show=False, title="Batch (Technical)")
    
    output_file = os.path.join(current_dir, "validation_batch_mixing.png")
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"✅ Plot saved to: {output_file}")

if __name__ == "__main__":
    main()