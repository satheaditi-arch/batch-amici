# src/amici/tools/synthetic.py

import numpy as np
import scanpy as sc

def make_semi_synthetic_batches(adata, seed=0):
    np.random.seed(seed)
    adata = adata.copy()
    xy = adata.obsm["spatial"].copy()

    mid_x = np.median(xy[:, 0])
    batch1_mask = xy[:, 0] <= mid_x
    batch2_mask = ~batch1_mask

    adata.obs["batch"] = np.where(batch1_mask, "Batch1", "Batch2")

    X = adata.X.astype(float)
    X[batch2_mask] *= 1.6

    gene_bias = np.random.normal(0, 0.3, adata.n_vars)
    X *= np.exp(gene_bias)

    dropout_mask = (np.random.rand(*X.shape) < 0.1) & batch2_mask[:, None]
    X[dropout_mask] = 0.0

    center = xy.mean(axis=0)
    radial = np.linalg.norm(xy - center, axis=1)
    high_radial = radial > np.percentile(radial, 60)
    X[high_radial] *= 0.6

    adata.X = X
    return adata
