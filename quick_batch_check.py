"""Visualize batch effects in semi-synthetic spatial data."""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("results/ba_amici_benchmark/data")
FIG_DIR = Path("results/ba_amici_benchmark/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Colors for batches and cell types
BATCH_COLORS = ['#E41A1C', '#377EB8', '#4DAF4A']  # Red, Blue, Green
CT_COLORS = ['#FF7F00', '#984EA3', '#A65628']     # Orange, Purple, Brown

def load_replicate(rep_idx):
    """Load replicate data directly with h5py."""
    rep_path = DATA_DIR / f"replicate_{rep_idx:02d}.h5ad"
    
    with h5py.File(rep_path, 'r') as f:
        # Expression matrix
        if 'X' in f and isinstance(f['X'], h5py.Dataset):
            X = f['X'][:]
        else:
            # Sparse matrix
            X_data = f['X']['data'][:]
            X_indices = f['X']['indices'][:]
            X_indptr = f['X']['indptr'][:]
            shape = tuple(f['X'].attrs['shape'])
            from scipy.sparse import csr_matrix
            X = csr_matrix((X_data, X_indices, X_indptr), shape=shape).toarray()
        
        # Spatial coordinates
        spatial = f['obsm']['spatial'][:]
        
        # Batch labels
        batch_codes = f['obs']['batch']['codes'][:]
        batch_categories = [x.decode() if isinstance(x, bytes) else x 
                          for x in f['obs']['batch']['categories'][:]]
        
        # Cell type labels
        ct_codes = f['obs']['cell_type']['codes'][:]
        ct_categories = [x.decode() if isinstance(x, bytes) else x 
                        for x in f['obs']['cell_type']['categories'][:]]
        
        # Subtype labels
        subtype_codes = f['obs']['subtype']['codes'][:]
        subtype_categories = [x.decode() if isinstance(x, bytes) else x 
                             for x in f['obs']['subtype']['categories'][:]]
    
    return {
        'X': X,
        'spatial': spatial,
        'batch_codes': batch_codes,
        'batch_names': [batch_categories[c] for c in batch_codes],
        'ct_codes': ct_codes,
        'ct_names': [ct_categories[c] for c in ct_codes],
        'subtype_codes': subtype_codes,
        'subtype_names': [subtype_categories[c] for c in subtype_codes],
    }


def plot_spatial_by_batch(data, rep_idx, ax=None):
    """Plot spatial coordinates colored by batch."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    spatial = data['spatial']
    batch_codes = data['batch_codes']
    
    for b in np.unique(batch_codes):
        mask = batch_codes == b
        ax.scatter(
            spatial[mask, 0], 
            spatial[mask, 1],
            c=BATCH_COLORS[b],
            s=1,
            alpha=0.5,
            label=f'Batch {b}'
        )
    
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title(f'Replicate {rep_idx}: Spatial Distribution by Batch')
    ax.legend(markerscale=5)
    ax.set_aspect('equal')
    
    return ax


def plot_spatial_by_celltype(data, rep_idx, ax=None):
    """Plot spatial coordinates colored by cell type."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    spatial = data['spatial']
    ct_codes = data['ct_codes']
    ct_names = np.unique(data['ct_names'])
    
    for i, ct in enumerate(sorted(ct_names)):
        mask = np.array(data['ct_names']) == ct
        ax.scatter(
            spatial[mask, 0], 
            spatial[mask, 1],
            c=CT_COLORS[i % len(CT_COLORS)],
            s=1,
            alpha=0.5,
            label=ct
        )
    
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title(f'Replicate {rep_idx}: Spatial Distribution by Cell Type')
    ax.legend(markerscale=5)
    ax.set_aspect('equal')
    
    return ax


def plot_pca_by_batch(data, rep_idx, ax=None):
    """Plot PCA colored by batch."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    X = data['X']
    batch_codes = data['batch_codes']
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    for b in np.unique(batch_codes):
        mask = batch_codes == b
        ax.scatter(
            X_pca[mask, 0], 
            X_pca[mask, 1],
            c=BATCH_COLORS[b],
            s=5,
            alpha=0.3,
            label=f'Batch {b}'
        )
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'Replicate {rep_idx}: PCA by Batch')
    ax.legend(markerscale=3)
    
    return ax, pca


def plot_pca_by_celltype(data, rep_idx, ax=None, pca=None):
    """Plot PCA colored by cell type."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    X = data['X']
    ct_names = np.unique(data['ct_names'])
    
    if pca is None:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
    else:
        X_pca = pca.transform(X)
    
    for i, ct in enumerate(sorted(ct_names)):
        mask = np.array(data['ct_names']) == ct
        ax.scatter(
            X_pca[mask, 0], 
            X_pca[mask, 1],
            c=CT_COLORS[i % len(CT_COLORS)],
            s=5,
            alpha=0.3,
            label=ct
        )
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'Replicate {rep_idx}: PCA by Cell Type')
    ax.legend(markerscale=3)
    
    return ax


def plot_batch_gene_profiles(data, rep_idx, n_genes=50):
    """Plot mean gene expression profiles per batch."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    X = data['X']
    batch_codes = data['batch_codes']
    n_batches = len(np.unique(batch_codes))
    
    # Compute batch means
    batch_means = []
    for b in range(n_batches):
        mask = batch_codes == b
        batch_means.append(X[mask].mean(axis=0))
    batch_means = np.array(batch_means)
    
    # Left: Heatmap of top variable genes across batches
    gene_vars = np.var(batch_means, axis=0)
    top_genes = np.argsort(gene_vars)[-n_genes:]
    
    im = axes[0].imshow(batch_means[:, top_genes], aspect='auto', cmap='viridis')
    axes[0].set_xlabel('Gene Index (top variable across batches)')
    axes[0].set_ylabel('Batch')
    axes[0].set_yticks(range(n_batches))
    axes[0].set_yticklabels([f'Batch {b}' for b in range(n_batches)])
    axes[0].set_title(f'Replicate {rep_idx}: Batch Mean Expression\n(Top {n_genes} batch-variable genes)')
    plt.colorbar(im, ax=axes[0], label='Mean Expression')
    
    # Right: Scatter plot comparing batch 0 vs batch 1 means
    axes[1].scatter(batch_means[0], batch_means[1], s=3, alpha=0.5)
    axes[1].plot([0, batch_means.max()], [0, batch_means.max()], 'r--', lw=1, label='y=x')
    
    # Compute correlation
    corr = np.corrcoef(batch_means[0], batch_means[1])[0, 1]
    axes[1].set_xlabel('Batch 0 Mean Expression')
    axes[1].set_ylabel('Batch 1 Mean Expression')
    axes[1].set_title(f'Replicate {rep_idx}: Batch 0 vs Batch 1\nCorrelation: {corr:.4f}')
    axes[1].legend()
    
    plt.tight_layout()
    return fig


def plot_batch_effect_summary(data, rep_idx):
    """Summary plot showing batch effect strength."""
    from sklearn.metrics import silhouette_score
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    X = data['X']
    batch_codes = data['batch_codes']
    spatial = data['spatial']
    
    # PCA
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X)
    
    # Row 1: Spatial plots
    # 1a. Spatial by batch
    for b in np.unique(batch_codes):
        mask = batch_codes == b
        axes[0, 0].scatter(spatial[mask, 0], spatial[mask, 1], 
                          c=BATCH_COLORS[b], s=1, alpha=0.3, label=f'Batch {b}')
    axes[0, 0].set_title('Spatial: Batch Distribution')
    axes[0, 0].legend(markerscale=5)
    axes[0, 0].set_aspect('equal')
    
    # 1b. Spatial by cell type
    ct_names = np.unique(data['ct_names'])
    for i, ct in enumerate(sorted(ct_names)):
        mask = np.array(data['ct_names']) == ct
        axes[0, 1].scatter(spatial[mask, 0], spatial[mask, 1],
                          c=CT_COLORS[i], s=1, alpha=0.3, label=ct)
    axes[0, 1].set_title('Spatial: Cell Type Distribution')
    axes[0, 1].legend(markerscale=5)
    axes[0, 1].set_aspect('equal')
    
    # 1c. Spatial by subtype (interaction status)
    subtype_names = data['subtype_names']
    unique_subtypes = np.unique(subtype_names)
    cmap = plt.cm.tab10
    for i, st in enumerate(unique_subtypes):
        mask = np.array(subtype_names) == st
        axes[0, 2].scatter(spatial[mask, 0], spatial[mask, 1],
                          c=[cmap(i)], s=1, alpha=0.3, label=st)
    axes[0, 2].set_title('Spatial: Subtype (Interaction Status)')
    axes[0, 2].legend(markerscale=5, fontsize=8)
    axes[0, 2].set_aspect('equal')
    
    # Row 2: Embedding plots
    # 2a. PCA by batch
    for b in np.unique(batch_codes):
        mask = batch_codes == b
        axes[1, 0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                          c=BATCH_COLORS[b], s=3, alpha=0.3, label=f'Batch {b}')
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1, 0].set_title('PCA: Batch Distribution')
    axes[1, 0].legend(markerscale=3)
    
    # 2b. PCA by cell type
    for i, ct in enumerate(sorted(ct_names)):
        mask = np.array(data['ct_names']) == ct
        axes[1, 1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                          c=CT_COLORS[i], s=3, alpha=0.3, label=ct)
    axes[1, 1].set_xlabel(f'PC1')
    axes[1, 1].set_ylabel(f'PC2')
    axes[1, 1].set_title('PCA: Cell Type Distribution')
    axes[1, 1].legend(markerscale=3)
    
    # 2c. Metrics summary
    sil_batch = silhouette_score(X_pca, batch_codes)
    sil_ct = silhouette_score(X_pca, data['ct_codes'])
    
    # Batch correlation
    batch_means = []
    for b in np.unique(batch_codes):
        mask = batch_codes == b
        batch_means.append(X[mask].mean(axis=0))
    corrs = []
    for i in range(len(batch_means)):
        for j in range(i+1, len(batch_means)):
            corrs.append(np.corrcoef(batch_means[i], batch_means[j])[0, 1])
    mean_corr = np.mean(corrs)
    
    # Variance explained by batch
    batch_var = np.var([X_pca[batch_codes == b].mean(0) for b in np.unique(batch_codes)], axis=0).sum()
    total_var = np.var(X_pca, axis=0).sum()
    var_explained = batch_var / total_var
    
    metrics_text = f"""
    BATCH EFFECT METRICS
    ════════════════════════════════════
    
    Batch Silhouette:     {sil_batch:+.4f}
      Target: > 0.10      {"✓ GOOD" if sil_batch > 0.1 else "✗ TOO WEAK"}
    
    Cell Type Silhouette: {sil_ct:+.4f}
      Target: > 0.20      {"✓ GOOD" if sil_ct > 0.2 else "⚠️ CHECK"}
    
    Batch Correlation:    {mean_corr:.4f}
      Target: < 0.95      {"✓ GOOD" if mean_corr < 0.95 else "✗ TOO SIMILAR"}
    
    Batch Var Explained:  {var_explained:.4f}
      Target: > 0.10      {"✓ GOOD" if var_explained > 0.1 else "✗ TOO WEAK"}
    
    ════════════════════════════════════
    """
    
    axes[1, 2].text(0.1, 0.5, metrics_text, transform=axes[1, 2].transAxes,
                   fontsize=11, verticalalignment='center', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Batch Effect Summary')
    
    plt.suptitle(f'Replicate {rep_idx}: Batch Effect Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig, {'sil_batch': sil_batch, 'sil_ct': sil_ct, 
                 'batch_corr': mean_corr, 'var_explained': var_explained}


def main():
    print("="*60)
    print("BATCH EFFECT VISUALIZATION")
    print("="*60)
    
    all_metrics = []
    
    for rep_idx in range(min(3, 10)):  # First 3 replicates
        rep_path = DATA_DIR / f"replicate_{rep_idx:02d}.h5ad"
        
        if not rep_path.exists():
            print(f"\nReplicate {rep_idx}: NOT FOUND")
            continue
        
        print(f"\nProcessing replicate {rep_idx}...")
        
        # Load data
        data = load_replicate(rep_idx)
        print(f"  Loaded {data['X'].shape[0]} cells, {data['X'].shape[1]} genes")
        
        # Generate summary figure
        fig, metrics = plot_batch_effect_summary(data, rep_idx)
        fig_path = FIG_DIR / f"replicate_{rep_idx:02d}_batch_effects.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path}")
        
        # Gene profile comparison
        fig = plot_batch_gene_profiles(data, rep_idx)
        fig_path = FIG_DIR / f"replicate_{rep_idx:02d}_gene_profiles.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path}")
        
        metrics['replicate'] = rep_idx
        all_metrics.append(metrics)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY ACROSS REPLICATES")
    print("="*60)
    print(f"{'Rep':<5} {'Batch Sil':<12} {'CT Sil':<12} {'Batch Corr':<12} {'Var Expl':<12}")
    print("-"*60)
    for m in all_metrics:
        status = "✓" if m['sil_batch'] > 0.1 and m['batch_corr'] < 0.95 else "⚠️"
        print(f"{m['replicate']:<5} {m['sil_batch']:<12.4f} {m['sil_ct']:<12.4f} "
              f"{m['batch_corr']:<12.4f} {m['var_explained']:<12.4f} {status}")
    
    print(f"\nFigures saved to: {FIG_DIR}")
    print("\nTARGETS:")
    print("  Batch Silhouette: > 0.10 (batches should be separable)")
    print("  Batch Correlation: < 0.95 (batches should differ)")
    print("  Variance Explained: > 0.10 (batch should explain variance)")


if __name__ == "__main__":
    main()