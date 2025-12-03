import numpy as np
import scanpy as sc

def make_pbmc_semi_synthetic(
    adata,
    seed=0,
    lib_scale=1.4,
    gene_bias_std=0.25,
    dropout_p=0.15,
    frac_genes_perturbed=0.3,
):
    """
    Create a semi-synthetic 2-batch PBMC dataset from a single PBMC AnnData.
    Batch2 has moderate technical distortions, but biology should still be visible.

    Parameters
    ----------
    adata : AnnData
        Single-batch PBMC data (preferably raw-ish counts or before heavy batch correction).
    lib_scale : float
        Multiplicative library size factor for Batch2.
    gene_bias_std : float
        Std of per-gene log-multiplicative noise (N(0, std)).
    dropout_p : float
        Extra dropout probability for Batch2.
    frac_genes_perturbed : float
        Fraction of genes to apply gene-wise bias to (not all genes).
    """
    np.random.seed(seed)
    adata = adata.copy()

    n_cells, n_genes = adata.shape

    # 1. Split cells into two pseudo-batches
    perm = np.random.permutation(n_cells)
    half = n_cells // 2
    idx_b1 = perm[:half]
    idx_b2 = perm[half:]

    batch = np.array(["Batch1"] * n_cells, dtype=object)
    batch[idx_b2] = "Batch2"
    adata.obs["batch"] = batch

    # Make sure X is dense float
    if not isinstance(adata.X, np.ndarray):
        X = adata.X.A.astype(float)
    else:
        X = adata.X.astype(float)

    # 2. Library size distortion for Batch2 (moderate)
    X[idx_b2, :] *= lib_scale

    # 3. Gene-wise multiplicative bias on a subset of genes
    n_perturb = int(frac_genes_perturbed * n_genes)
    gene_idx = np.random.choice(n_genes, size=n_perturb, replace=False)

    gene_bias = np.random.normal(0.0, gene_bias_std, size=n_perturb)  # log-scale
    gene_scale = np.exp(gene_bias)  # multiplicative factors

    # apply only to selected genes for both batches
    X[:, gene_idx] *= gene_scale

    # 4. Extra dropout for Batch2
    mask = np.random.rand(len(idx_b2), n_genes) < dropout_p
    X[idx_b2, :][mask] = 0.0

    adata.X = X
    return adata
