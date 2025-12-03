# benchmarks/xenium_mouse/preprocess_xenium_mouse.py

import scanpy as sc
import pandas as pd
import numpy as np
import os

# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "xenium_mouse")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# Preprocess One Xenium Replicate
# =========================================================

def preprocess_rep(rep_name):
    print(f"\n Processing {rep_name}...")

    rep_dir = os.path.join(RAW_DIR, rep_name)

    expr_path = os.path.join(rep_dir, "cell_feature_matrix.h5")
    cells_path = os.path.join(rep_dir, "cells.csv.gz")

    # -------------------------------
    # 1. Load expression
    # -------------------------------
    adata = sc.read_10x_h5(expr_path)
    adata.var_names_make_unique()

    print("   Expression loaded:", adata.shape)

    # -------------------------------
    # 2. Load cell metadata
    # -------------------------------
    cells = pd.read_csv(cells_path)

    # Try all common Xenium coordinate names
    possible_keys = [
        ("x_centroid", "y_centroid"),
        ("center_x", "center_y"),
        ("x", "y"),
        ("cell_x", "cell_y"),
    ]

    spatial = None
    for xk, yk in possible_keys:
        if xk in cells.columns and yk in cells.columns:
            spatial = cells[[xk, yk]].values
            print(f"    Found spatial coords in columns: {xk}, {yk}")
            break

    if spatial is None:
        print(" Available columns in cells.csv.gz:")
        print(list(cells.columns))
        raise ValueError(" Could not find spatial centroid columns in cells.csv.gz")

    # -------------------------------
    # 3. Align metadata with expression
    # -------------------------------
    if len(spatial) != adata.n_obs:
        print(f" Mismatch: cells metadata = {len(spatial)}, expression = {adata.n_obs}")
        raise ValueError("Cell metadata and expression cell counts do not match.")

    adata.obsm["spatial"] = spatial

    # -------------------------------
    # 4. Basic QC + normalization
    # -------------------------------
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # -------------------------------
    # 5. Save
    # -------------------------------
    out_path = os.path.join(OUT_DIR, f"xenium_{rep_name}_preprocessed.h5ad")
    adata.write(out_path)

    print("    Saved to:", out_path)


# =========================================================
# Run for All Replicates
# =========================================================

if __name__ == "__main__":
    preprocess_rep("Rep1")
    preprocess_rep("Rep2")
    preprocess_rep("Rep3")

    print("\n All Xenium replicates preprocessed successfully.")
