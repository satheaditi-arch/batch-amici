import sys
import os
import torch
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
sys.path.append(src_path)

from amici import AMICI

def load_and_embed(run_id, data_path, label="Model"):
    print(f"--- Processing {label} ({run_id}) ---")
    model_path = os.path.join(current_dir, "results", run_id)
    
    # Load Data
    adata = sc.read_h5ad(data_path)
    if "spatial" not in adata.obsm:
        if "x" in adata.obs and "y" in adata.obs:
            adata.obsm["spatial"] = np.column_stack((adata.obs["x"].values, adata.obs["y"].values))
            
    # IMPORTANT: For Baseline, we must blind the data again so it matches training
    if "baseline" in run_id:
        adata.obs['batch'] = 'batch_0'
        adata.obs['batch'] = adata.obs['batch'].astype('category')

    # Setup & Load
    AMICI.setup_anndata(adata, labels_key="cell_type", batch_key="batch", coord_obsm_key="spatial", n_neighbors=30)
    model = AMICI.load(model_path, adata=adata)
    
    # Get Residuals
    residuals = model.get_predictions(adata, batch_size=128, get_residuals=True)
    adata.obsm["X_emb"] = residuals
    
    # Restore real batch labels for plotting (Crucial for Baseline!)
    if "baseline" in run_id:
        # Reload original batch labels so we can see if they separated
        adata_orig = sc.read_h5ad(data_path)
        adata.obs['real_batch'] = adata_orig.obs['batch'].values
    else:
        adata.obs['real_batch'] = adata.obs['batch']

    # UMAP
    sc.pp.neighbors(adata, use_rep="X_emb")
    sc.tl.umap(adata)
    return adata

def main():
    DATA_PATH = os.path.join(current_dir, "ba_amici_benchmark", "replicate_00.h5ad")
    
    # 1. Process Both Models
    # Note: Ensure you ran train_baseline.py first!
    adata_base = load_and_embed("baseline_amici_run", DATA_PATH, "Baseline")
    adata_ba = load_and_embed("ba_amici_test_run", DATA_PATH, "BA-AMICI")
    
    # 2. Plotting
    print("Generating Comparison Plot...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Row 1: Baseline (BEFORE)
    sc.pl.umap(adata_base, color="cell_type", ax=axes[0,0], show=False, title="Baseline: Biology (Should Cluster)")
    sc.pl.umap(adata_base, color="real_batch", ax=axes[0,1], show=False, title="Baseline: Batches (Should Separate)")
    
    # Row 2: BA-AMICI (AFTER)
    sc.pl.umap(adata_ba, color="cell_type", ax=axes[1,0], show=False, title="BA-AMICI: Biology (Should Cluster)")
    sc.pl.umap(adata_ba, color="real_batch", ax=axes[1,1], show=False, title="BA-AMICI: Batches (Should Mix!)")
    
    plt.tight_layout()
    save_path = "comparison_plot.png"
    plt.savefig(save_path)
    print(f"✅ Comparison saved to {save_path}")

if __name__ == "__main__":
    main()