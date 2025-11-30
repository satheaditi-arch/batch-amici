import sys
import os
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Config
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(current_dir, "ba_amici_benchmark", "replicate_00.h5ad")

def main():
    print(f"--- INSPECTING DATASET: {DATA_PATH} ---")
    
    if not os.path.exists(DATA_PATH):
        print("❌ ERROR: File not found.")
        return

    adata = sc.read_h5ad(DATA_PATH)
    
    # 1. Check Batch Labels
    print("\n1. Checking Batch Labels:")
    if "batch" in adata.obs:
        counts = adata.obs['batch'].value_counts()
        print(counts)
        if len(counts) < 2:
            print("❌ WARNING: Only 1 batch found! Batch correction is impossible.")
        else:
            print("✅ Batch labels look correct.")
    else:
        print("❌ ERROR: No 'batch' column found in adata.obs")

    # 2. Check Gene Statistics (Did the shift work?)
    print("\n2. Checking Gene Expression Statistics:")
    if isinstance(adata.X, np.ndarray):
        X = adata.X
    else:
        X = adata.X.toarray()
        
    # Calculate mean expression per batch
    for batch in adata.obs['batch'].unique():
        mask = adata.obs['batch'] == batch
        mean_expr = np.mean(X[mask], axis=0)
        print(f"   Batch {batch} - Global Mean Expression: {np.mean(mean_expr):.4f}")

    # 3. Calculate Silhouette Score (How separated are they?)
    # High Score (close to 1.0) = Batches are completely separated islands (Hard task)
    # Low Score (close to 0.0) = Batches are mixed (Easy task)
    print("\n3. Calculating Raw Silhouette Score (Batch Separation)...")
    sc.pp.pca(adata)
    sil = silhouette_score(adata.obsm['X_pca'], adata.obs['batch'])
    print(f"   Raw Batch Silhouette Score: {sil:.4f}")
    
    if sil > 0.2:
        print("   -> Batches are STRONGLY separated. (The 'Hard' simulation worked).")
    elif sil < 0.05:
        print("   -> Batches are already mixed. (The simulation is too weak).")

    # 4. Visualize Raw Data
    print("\n4. Generating Raw UMAP...")
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sc.pl.umap(adata, color="cell_type", ax=ax[0], show=False, title="Raw: Biology")
    sc.pl.umap(adata, color="batch", ax=ax[1], show=False, title=f"Raw: Batches (Sil={sil:.2f})")
    
    plt.tight_layout()
    plt.savefig("dataset_inspection.png")
    print("✅ Saved 'dataset_inspection.png'. Check this image!")

if __name__ == "__main__":
    main()