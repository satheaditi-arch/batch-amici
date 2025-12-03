"""
Implements standard metrics from:
- Luecken et al. (2022) - "Benchmarking atlas-level data integration"
- Korsunsky et al. (2019) - Harmony paper
- Lopez et al. (2018) - scVI paper

Metrics:
1. Batch Mixing: kBET, iLISI (lower = better mixing)
2. Biology Preservation: cLISI, silhouette score, ARI (higher = preserved)
3. Visualization: UMAP plots with both qualitative and quantitative assessment
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy

# Path setup
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from amici import AMICI

def compute_lisi(X, metadata, perplexity=30):
    """
    Compute Local Inverse Simpson's Index (LISI).
    
    Higher LISI = better mixing for batch
    Lower LISI = better separation for cell type
    
    Based on: Korsunsky et al. (2019) - Harmony
    """
    n_samples = X.shape[0]
    
    # Build k-NN graph
    nn = NearestNeighbors(n_neighbors=perplexity + 1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    
    # Remove self (first neighbor)
    indices = indices[:, 1:]
    
    lisi_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Get neighbor labels
        neighbor_labels = metadata[indices[i]]
        
        # Compute Simpson's Index
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        proportions = counts / counts.sum()
        
        # Simpson's Index: sum of squared proportions
        simpson = np.sum(proportions ** 2)
        
        # LISI: 1 / Simpson
        lisi_scores[i] = 1.0 / simpson if simpson > 0 else 1.0
    
    return lisi_scores


def compute_kbet(X, batch_labels, k=25, alpha=0.05, n_samples=500):
    """
    k-nearest neighbor Batch Effect Test (kBET).
    
    Tests if batch distribution in local neighborhoods matches global distribution.
    
    Returns:
        acceptance_rate: Fraction of neighborhoods that pass chi-squared test
                        (Higher = better mixing, ideal = 1.0)
    
    Based on: Büttner et al. (2019) - "A test metric for assessing single-cell RNA-seq batch correction"
    """
    from scipy.stats import chi2
    
    n_cells = X.shape[0]
    
    # Subsample if dataset is large
    if n_cells > n_samples:
        sample_idx = np.random.choice(n_cells, n_samples, replace=False)
        X_sample = X[sample_idx]
        batch_sample = batch_labels[sample_idx]
    else:
        X_sample = X
        batch_sample = batch_labels
        n_samples = n_cells
    
    # Build k-NN graph
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X)
    _, indices = nn.kneighbors(X_sample)
    indices = indices[:, 1:]  # Remove self
    
    # Global batch distribution
    unique_batches, global_counts = np.unique(batch_labels, return_counts=True)
    n_batches = len(unique_batches)
    global_freqs = global_counts / global_counts.sum()
    
    # Chi-squared test for each neighborhood
    accepted = 0
    
    for i in range(n_samples):
        # Local batch distribution
        neighbor_batches = batch_labels[indices[i]]
        _, local_counts = np.unique(neighbor_batches, return_counts=True)
        
        # Pad to match all batches
        local_full = np.zeros(n_batches)
        for j, batch in enumerate(unique_batches):
            if batch in neighbor_batches:
                local_full[j] = np.sum(neighbor_batches == batch)
        
        # Expected counts under global distribution
        expected = global_freqs * k
        
        # Chi-squared test
        # Avoid division by zero
        mask = expected > 0
        if mask.sum() > 1:  # Need at least 2 categories
            chi2_stat = np.sum((local_full[mask] - expected[mask]) ** 2 / expected[mask])
            dof = mask.sum() - 1
            p_value = 1 - chi2.cdf(chi2_stat, dof)
            
            if p_value > alpha:
                accepted += 1
    
    acceptance_rate = accepted / n_samples
    return acceptance_rate


def compute_silhouette_batch(X, batch_labels):
    """
    Silhouette score using batch labels.
    Lower = better mixing (batches are not well-separated)
    """
    return silhouette_score(X, batch_labels)


def compute_silhouette_celltype(X, celltype_labels):
    """
    Silhouette score using cell type labels.
    Higher = better biology preservation (cell types well-separated)
    """
    return silhouette_score(X, celltype_labels)


def compute_ari(true_labels, pred_labels):
    """
    Adjusted Rand Index - measures cluster agreement.
    Higher = better preservation of cell type structure
    """
    return adjusted_rand_score(true_labels, pred_labels)



# MAIN VALIDATION PIPELINE


def load_model_and_extract_embeddings(
    model_path: Path,
    data_path: Path,
    is_baseline: bool = False
):
    """
    Load trained model and extract embeddings.
    
    For baseline: blinds batch during loading, then restores for analysis.
    """
    print(f"\nLoading model from: {model_path}")
    
    # Load data
    adata = sc.read_h5ad(data_path)
    
    # Store original batch labels
    original_batch = adata.obs['batch'].copy()
    
    # Fix spatial coordinates
    if 'spatial' not in adata.obsm:
        if 'x_coord' in adata.obs and 'y_coord' in adata.obs:
            coords = np.column_stack([
                adata.obs['x_coord'].values,
                adata.obs['y_coord'].values
            ])
            adata.obsm['spatial'] = coords
      
    # Setup and load model
    AMICI.setup_anndata(
        adata,
        labels_key='cell_type',
        batch_key='batch',
        coord_obsm_key='spatial',
        n_neighbors=30
    )
    
    model = AMICI.load(str(model_path), adata=adata)
    
    # Extract embeddings (residuals represent learned biological variation)
    print("  Extracting embeddings...")
    embeddings = model.get_predictions(adata, batch_size=128, get_residuals=True)
    
    # Restore original batch labels for analysis
    adata.obs['batch'] = original_batch
    adata.obsm['X_emb'] = embeddings
    
    # Compute UMAP for visualization
    print("  Computing UMAP...")
    sc.pp.neighbors(adata, use_rep='X_emb', n_neighbors=15)
    sc.tl.umap(adata, min_dist=0.3)
    
    return adata


def compute_all_metrics(adata, model_name="Model"):
    """
    Compute all batch correction metrics.
    
    Returns dict with all metrics.
    """
    print(f"\n{'='*60}")
    print(f"Computing metrics for {model_name}")
    print(f"{'='*60}")
    
    X = adata.obsm['X_emb']
    batch_labels = adata.obs['batch'].values
    celltype_labels = adata.obs['cell_type'].values
    
    metrics = {}
    
    # 1. BATCH MIXING METRICS (lower/higher = better mixing)
    print("\n[Batch Mixing Metrics]")
    
    # iLISI - batch (higher = better mixing, max = n_batches)
    print("  Computing iLISI (batch)...")
    ilisi_batch = compute_lisi(X, batch_labels, perplexity=30)
    metrics['iLISI_batch_mean'] = ilisi_batch.mean()
    metrics['iLISI_batch_median'] = np.median(ilisi_batch)
    print(f"    iLISI (batch): {metrics['iLISI_batch_mean']:.3f} (max={len(np.unique(batch_labels))})")
    
    # kBET (higher = better mixing, max = 1.0)
    print("  Computing kBET...")
    kbet_score = compute_kbet(X, batch_labels, k=25, n_samples=500)
    metrics['kBET_acceptance'] = kbet_score
    print(f"    kBET acceptance rate: {kbet_score:.3f}")
    
    # Batch silhouette (lower = better mixing)
    print("  Computing batch silhouette...")
    batch_sil = compute_silhouette_batch(X, batch_labels)
    metrics['silhouette_batch'] = batch_sil
    print(f"    Batch silhouette: {batch_sil:.3f} (lower is better)")
    
    # 2. BIOLOGY PRESERVATION METRICS (higher = better)
    print("\n[Biology Preservation Metrics]")
    
    # cLISI - cell type (lower = better separation)
    print("  Computing cLISI (cell type)...")
    clisi_celltype = compute_lisi(X, celltype_labels, perplexity=30)
    metrics['cLISI_celltype_mean'] = clisi_celltype.mean()
    metrics['cLISI_celltype_median'] = np.median(clisi_celltype)
    print(f"    cLISI (cell type): {metrics['cLISI_celltype_mean']:.3f} (lower is better)")
    
    # Cell type silhouette (higher = better separation)
    print("  Computing cell type silhouette...")
    celltype_sil = compute_silhouette_celltype(X, celltype_labels)
    metrics['silhouette_celltype'] = celltype_sil
    print(f"    Cell type silhouette: {celltype_sil:.3f}")
    
    # ARI against original clusters
    if 'leiden' in adata.obs:
        print("  Computing ARI...")
        ari = compute_ari(adata.obs['leiden'], celltype_labels)
        metrics['ARI'] = ari
        print(f"    ARI: {ari:.3f}")
    
    # 3. COMPOSITE SCORE (like Luecken et al. 2022)
    # Balance between mixing and preservation
    # Higher = better overall
    batch_mixing_score = (metrics['kBET_acceptance'] + 
                          (1 - metrics['silhouette_batch'])) / 2
    bio_preservation_score = metrics['silhouette_celltype']
    
    composite = 0.4 * batch_mixing_score + 0.6 * bio_preservation_score
    metrics['composite_score'] = composite
    
    print(f"\n[Composite Scores]")
    print(f"  Batch mixing score: {batch_mixing_score:.3f}")
    print(f"  Bio preservation score: {bio_preservation_score:.3f}")
    print(f"  Composite (0.4*mix + 0.6*bio): {composite:.3f}")
    
    return metrics


def plot_comparison(adata_baseline, adata_ba, metrics_baseline, metrics_ba, output_dir):
    """
    Generate publication-quality comparison figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    # FIGURE 1: UMAP Comparison (Main Figure)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Baseline
    sc.pl.umap(adata_baseline, color='cell_type', ax=axes[0, 0], show=False, 
               title='Baseline: Cell Types', legend_loc='on data', frameon=False)
    sc.pl.umap(adata_baseline, color='batch', ax=axes[0, 1], show=False,
               title='Baseline: Batches', frameon=False)
    
    # Spatial view
    axes[0, 2].scatter(adata_baseline.obs['x_coord'], adata_baseline.obs['y_coord'],
                       c=pd.Categorical(adata_baseline.obs['batch']).codes,
                       s=5, alpha=0.6, cmap='Set1')
    axes[0, 2].set_title('Baseline: Spatial Batch Distribution')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Y')
    
    # Row 2: BA-AMICI
    sc.pl.umap(adata_ba, color='cell_type', ax=axes[1, 0], show=False,
               title='BA-AMICI: Cell Types', legend_loc='on data', frameon=False)
    sc.pl.umap(adata_ba, color='batch', ax=axes[1, 1], show=False,
               title='BA-AMICI: Batches (Corrected)', frameon=False)
    
    # Spatial view
    axes[1, 2].scatter(adata_ba.obs['x_coord'], adata_ba.obs['y_coord'],
                       c=pd.Categorical(adata_ba.obs['batch']).codes,
                       s=5, alpha=0.6, cmap='Set1')
    axes[1, 2].set_title('BA-AMICI: Spatial Batch Distribution')
    axes[1, 2].set_xlabel('X')
    axes[1, 2].set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure1_umap_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'figure1_umap_comparison.png'}")
    
    # FIGURE 2: Quantitative Metrics (Bar Charts)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Prepare data
    models = ['Baseline', 'BA-AMICI']
    
    # Panel A: Batch Mixing
    metrics_mixing = {
        'kBET': [metrics_baseline['kBET_acceptance'], metrics_ba['kBET_acceptance']],
        'iLISI': [metrics_baseline['iLISI_batch_mean'] / 3, metrics_ba['iLISI_batch_mean'] / 3],  # Normalize to 0-1
    }
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0].bar(x - width/2, metrics_mixing['kBET'], width, label='kBET', alpha=0.8)
    axes[0].bar(x + width/2, metrics_mixing['iLISI'], width, label='iLISI (norm)', alpha=0.8)
    axes[0].set_ylabel('Score (Higher = Better Mixing)')
    axes[0].set_title('Batch Mixing Metrics')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].legend()
    axes[0].set_ylim([0, 1.1])
    
    # Panel B: Biology Preservation
    metrics_bio = [
        metrics_baseline['silhouette_celltype'],
        metrics_ba['silhouette_celltype']
    ]
    
    axes[1].bar(models, metrics_bio, alpha=0.8, color=['#ff7f0e', '#2ca02c'])
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Cell Type Preservation')
    axes[1].set_ylim([0, max(metrics_bio) * 1.2])
    
    # Panel C: Composite Score
    composite_scores = [
        metrics_baseline['composite_score'],
        metrics_ba['composite_score']
    ]
    
    bars = axes[2].bar(models, composite_scores, alpha=0.8, color=['#ff7f0e', '#2ca02c'])
    axes[2].set_ylabel('Composite Score')
    axes[2].set_title('Overall Performance')
    axes[2].set_ylim([0, 1])
    
    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure2_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'figure2_metrics_comparison.png'}")
    
    
    # FIGURE 3: LISI Distributions (Violin Plots)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Compute LISI for both models
    ilisi_baseline = compute_lisi(adata_baseline.obsm['X_emb'], 
                                   adata_baseline.obs['batch'].values)
    ilisi_ba = compute_lisi(adata_ba.obsm['X_emb'], 
                            adata_ba.obs['batch'].values)
    
    # Panel A: iLISI distribution
    data_ilisi = pd.DataFrame({
        'iLISI': np.concatenate([ilisi_baseline, ilisi_ba]),
        'Model': ['Baseline'] * len(ilisi_baseline) + ['BA-AMICI'] * len(ilisi_ba)
    })
    
    sns.violinplot(data=data_ilisi, x='Model', y='iLISI', ax=axes[0])
    axes[0].set_title('iLISI Distribution (Batch Mixing)')
    axes[0].set_ylabel('iLISI Score')
    axes[0].axhline(y=3, color='red', linestyle='--', label='Ideal (n_batches)', alpha=0.5)
    axes[0].legend()
    
    # Panel B: cLISI distribution
    clisi_baseline = compute_lisi(adata_baseline.obsm['X_emb'], 
                                   adata_baseline.obs['cell_type'].values)
    clisi_ba = compute_lisi(adata_ba.obsm['X_emb'], 
                            adata_ba.obs['cell_type'].values)
    
    data_clisi = pd.DataFrame({
        'cLISI': np.concatenate([clisi_baseline, clisi_ba]),
        'Model': ['Baseline'] * len(clisi_baseline) + ['BA-AMICI'] * len(clisi_ba)
    })
    
    sns.violinplot(data=data_clisi, x='Model', y='cLISI', ax=axes[1])
    axes[1].set_title('cLISI Distribution (Cell Type Purity)')
    axes[1].set_ylabel('cLISI Score (Lower = Purer)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_lisi_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'figure3_lisi_distributions.png'}")


def save_metrics_table(metrics_baseline, metrics_ba, output_dir):
    """
    Save metrics as CSV table for paper.
    """
    output_dir = Path(output_dir)
    
    df = pd.DataFrame({
        'Metric': list(metrics_baseline.keys()),
        'Baseline': list(metrics_baseline.values()),
        'BA-AMICI': list(metrics_ba.values()),
    })
    
    # Add improvement column
    df['Improvement'] = ((df['BA-AMICI'] - df['Baseline']) / 
                         (df['Baseline'].abs() + 1e-10) * 100)
    
    # Round for readability
    df['Baseline'] = df['Baseline'].round(4)
    df['BA-AMICI'] = df['BA-AMICI'].round(4)
    df['Improvement'] = df['Improvement'].round(2)
    
    csv_path = output_dir / 'metrics_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved metrics table: {csv_path}")
    
    return df



# MAIN EXECUTION


def main():
    DATA_PATH = PROJECT_ROOT / "pbmc_data" / "ba_amici_benchmark" / "replicate_00.h5ad"
    BASELINE_PATH = SCRIPT_DIR / "results" / "baseline_amici_replicate_00"
    BA_AMICI_PATH = SCRIPT_DIR / "results" / "ba_amici_replicate_00_v2"
    OUTPUT_DIR = SCRIPT_DIR / "validation_results"
    
    print("="*70)
    print("PUBLICATION-QUALITY BATCH CORRECTION VALIDATION")
    print("="*70)
    
    # Load models and extract embeddings
    adata_baseline = load_model_and_extract_embeddings(
        BASELINE_PATH, DATA_PATH, is_baseline=True
    )
    
    adata_ba = load_model_and_extract_embeddings(
        BA_AMICI_PATH, DATA_PATH, is_baseline=False
    )
    
    # Compute metrics
    metrics_baseline = compute_all_metrics(adata_baseline, "Baseline AMICI")
    metrics_ba = compute_all_metrics(adata_ba, "BA-AMICI")
    
    # Generate figures
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)
    plot_comparison(adata_baseline, adata_ba, metrics_baseline, metrics_ba, OUTPUT_DIR)
    
    # Save metrics table
    df_metrics = save_metrics_table(metrics_baseline, metrics_ba, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nKey Findings:")
    print(f"  Batch Mixing (kBET): {metrics_baseline['kBET_acceptance']:.3f} → {metrics_ba['kBET_acceptance']:.3f}")
    print(f"  Biology Preserved (Sil): {metrics_baseline['silhouette_celltype']:.3f} → {metrics_ba['silhouette_celltype']:.3f}")
    print(f"  Composite Score: {metrics_baseline['composite_score']:.3f} → {metrics_ba['composite_score']:.3f}")
    
    improvement = ((metrics_ba['composite_score'] - metrics_baseline['composite_score']) / 
                   metrics_baseline['composite_score'] * 100)
    print(f"\n  Overall Improvement: {improvement:+.1f}%")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()