"""
BA-AMICI Validation Script - FIXED VERSION

Properly evaluates batch correction with:
1. Correct iLISI/cLISI computation
2. Clear interpretation of metrics
3. Diagnostic checks for data quality

Key Metrics:
- iLISI (batch): Higher = better mixing (ideal = n_batches)
- cLISI (cell type): Lower = better purity (ideal = 1.0)
- kBET: Higher = better mixing (ideal = 1.0)
- Silhouette batch: Lower/negative = better mixing
- Silhouette celltype: Higher = better preservation
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
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')

# Path setup
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from amici import AMICI


# ==============================================================================
# METRIC COMPUTATION
# ==============================================================================

def compute_lisi(X, labels, perplexity=30):
    """
    Compute Local Inverse Simpson's Index (LISI).
    
    LISI measures the effective number of categories in local neighborhoods.
    
    For BATCH labels (iLISI):
        - Higher = better mixing
        - Ideal = n_batches (all batches equally represented)
        - Min = 1.0 (all neighbors same batch)
    
    For CELL TYPE labels (cLISI):
        - Lower = better purity  
        - Ideal = 1.0 (all neighbors same type)
        - Max = n_celltypes (random mixing)
    """
    n_samples = X.shape[0]
    k = min(perplexity, n_samples - 1)
    
    # Build k-NN graph
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree')
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    
    # Remove self (first neighbor)
    indices = indices[:, 1:]
    
    lisi_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        neighbor_labels = labels[indices[i]]
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        proportions = counts / counts.sum()
        
        # Simpson's Index: sum of squared proportions
        simpson = np.sum(proportions ** 2)
        
        # LISI: 1 / Simpson (effective number of categories)
        lisi_scores[i] = 1.0 / simpson if simpson > 0 else 1.0
    
    return lisi_scores


def compute_kbet(X, batch_labels, k=25, alpha=0.05, n_samples=1000):
    """
    k-nearest neighbor Batch Effect Test (kBET).
    
    Tests if local batch distribution matches global distribution.
    
    Returns:
        acceptance_rate: Higher = better mixing (ideal = 1.0)
    """
    n_cells = X.shape[0]
    
    # Subsample for speed
    if n_cells > n_samples:
        sample_idx = np.random.choice(n_cells, n_samples, replace=False)
        X_sample = X[sample_idx]
    else:
        sample_idx = np.arange(n_cells)
        X_sample = X
        n_samples = n_cells
    
    # Build k-NN on full data
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree')
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
        neighbor_batches = batch_labels[indices[i]]
        
        # Count each batch in neighborhood
        local_counts = np.zeros(n_batches)
        for j, batch in enumerate(unique_batches):
            local_counts[j] = np.sum(neighbor_batches == batch)
        
        # Expected counts
        expected = global_freqs * k
        
        # Chi-squared test
        mask = expected > 0
        if mask.sum() > 1:
            chi2_stat = np.sum((local_counts[mask] - expected[mask]) ** 2 / expected[mask])
            dof = mask.sum() - 1
            p_value = 1 - chi2.cdf(chi2_stat, dof)
            
            if p_value > alpha:
                accepted += 1
    
    return accepted / n_samples


def compute_all_metrics(X_embed, batch_labels, celltype_labels, model_name="Model"):
    """
    Compute comprehensive batch correction metrics.
    """
    print(f"\n{'='*60}")
    print(f"Computing metrics for: {model_name}")
    print(f"{'='*60}")
    
    n_batches = len(np.unique(batch_labels))
    n_celltypes = len(np.unique(celltype_labels))
    
    metrics = {}
    
    # ===== BATCH MIXING METRICS (higher = better) =====
    print("\n[Batch Mixing - Higher is Better]")
    
    # iLISI
    print("  Computing iLISI...")
    ilisi = compute_lisi(X_embed, batch_labels, perplexity=30)
    metrics['iLISI_mean'] = ilisi.mean()
    metrics['iLISI_median'] = np.median(ilisi)
    # Normalize to [0, 1] where 1 = perfect
    metrics['iLISI_normalized'] = (ilisi.mean() - 1) / (n_batches - 1)
    print(f"    iLISI: {metrics['iLISI_mean']:.3f} (ideal={n_batches}, normalized={metrics['iLISI_normalized']:.3f})")
    
    # kBET
    print("  Computing kBET...")
    metrics['kBET_acceptance'] = compute_kbet(X_embed, batch_labels, k=25)
    print(f"    kBET acceptance: {metrics['kBET_acceptance']:.3f} (ideal=1.0)")
    
    # Silhouette (batch) - want NEGATIVE
    print("  Computing batch silhouette...")
    metrics['silhouette_batch'] = silhouette_score(X_embed, batch_labels)
    print(f"    Batch silhouette: {metrics['silhouette_batch']:.3f} (ideal<0)")
    
    # ===== BIOLOGY PRESERVATION METRICS =====
    print("\n[Biology Preservation]")
    
    # cLISI - want LOW
    print("  Computing cLISI...")
    clisi = compute_lisi(X_embed, celltype_labels, perplexity=30)
    metrics['cLISI_mean'] = clisi.mean()
    metrics['cLISI_median'] = np.median(clisi)
    # Normalize: 1 = perfect purity, 0 = random
    metrics['cLISI_normalized'] = 1 - (clisi.mean() - 1) / (n_celltypes - 1)
    print(f"    cLISI: {metrics['cLISI_mean']:.3f} (ideal=1.0, normalized={metrics['cLISI_normalized']:.3f})")
    
    # Silhouette (celltype) - want HIGH
    print("  Computing celltype silhouette...")
    metrics['silhouette_celltype'] = silhouette_score(X_embed, celltype_labels)
    print(f"    Celltype silhouette: {metrics['silhouette_celltype']:.3f} (ideal=1.0)")
    
    # ===== COMPOSITE SCORE =====
    # Balance between batch mixing and biology preservation
    batch_score = (
        0.5 * metrics['iLISI_normalized'] +
        0.3 * metrics['kBET_acceptance'] +
        0.2 * (1 - (metrics['silhouette_batch'] + 1) / 2)  # Convert to [0,1] where higher=better
    )
    bio_score = (
        0.5 * metrics['cLISI_normalized'] +
        0.5 * (metrics['silhouette_celltype'] + 1) / 2  # Convert to [0,1]
    )
    metrics['batch_mixing_score'] = batch_score
    metrics['bio_preservation_score'] = bio_score
    metrics['composite_score'] = 0.5 * batch_score + 0.5 * bio_score
    
    print(f"\n[Summary Scores]")
    print(f"    Batch mixing: {batch_score:.3f}")
    print(f"    Bio preservation: {bio_score:.3f}")
    print(f"    COMPOSITE: {metrics['composite_score']:.3f}")
    
    return metrics


# ==============================================================================
# DATA QUALITY CHECKS
# ==============================================================================

def diagnose_data_quality(adata):
    """
    Check if the data has sufficient batch effects for meaningful evaluation.
    """
    print("\n" + "="*60)
    print("DATA QUALITY DIAGNOSTICS")
    print("="*60)
    
    # Check batch distribution
    batch_counts = adata.obs['batch'].value_counts()
    print("\nBatch distribution:")
    for batch, count in batch_counts.items():
        print(f"  {batch}: {count} cells ({count/adata.n_obs*100:.1f}%)")
    
    # Check cell type distribution per batch
    print("\nCell type distribution per batch:")
    ct_by_batch = pd.crosstab(adata.obs['batch'], adata.obs['cell_type'], normalize='index')
    print(ct_by_batch.round(3))
    
    # Compute batch separation in raw space
    print("\nBatch separation in raw expression space:")
    if 'X_pca' not in adata.obsm:
        sc.tl.pca(adata, n_comps=50)
    
    X_pca = adata.obsm['X_pca'][:, :30]
    batch_labels = adata.obs['batch'].values
    
    ilisi_raw = compute_lisi(X_pca, batch_labels, perplexity=30)
    sil_batch_raw = silhouette_score(X_pca[:1000], batch_labels[:1000])
    
    n_batches = len(batch_counts)
    print(f"  iLISI (raw): {ilisi_raw.mean():.3f} (ideal for NO batch effect = {n_batches})")
    print(f"  Silhouette (raw): {sil_batch_raw:.3f} (positive = batches separate)")
    
    # Interpretation
    if ilisi_raw.mean() > n_batches * 0.8:
        print("\n⚠️  WARNING: Batches are already well-mixed in raw data!")
        print("   This means batch correction may have little effect.")
        print("   Consider generating data with stronger batch effects.")
    elif sil_batch_raw < 0.1:
        print("\n⚠️  WARNING: Batches don't separate clearly in raw data!")
        print("   Batch effects may be too weak for meaningful evaluation.")
    else:
        print("\n✓ Data appears suitable for batch correction evaluation.")
    
    return {
        'ilisi_raw': ilisi_raw.mean(),
        'silhouette_raw': sil_batch_raw,
        'n_batches': n_batches,
    }


# ==============================================================================
# MODEL LOADING & EMBEDDING EXTRACTION
# ==============================================================================

def load_model_and_extract(model_path, data_path, model_name="Model"):
    """
    Load model and extract embeddings for evaluation.
    """
    print(f"\n{'='*60}")
    print(f"Loading {model_name}")
    print(f"{'='*60}")
    print(f"  Model: {model_path}")
    print(f"  Data:  {data_path}")
    
    # Load data
    adata = sc.read_h5ad(data_path)
    
    # Fix spatial coordinates
    if 'spatial' not in adata.obsm:
        if 'x_coord' in adata.obs and 'y_coord' in adata.obs:
            adata.obsm['spatial'] = np.column_stack([
                adata.obs['x_coord'].values,
                adata.obs['y_coord'].values
            ])
    
    # Setup and load model
    AMICI.setup_anndata(
        adata,
        labels_key='cell_type',
        batch_key='batch',
        coord_obsm_key='spatial',
        n_neighbors=30
    )
    
    model = AMICI.load(str(model_path), adata=adata)
    
    # Extract embeddings
    print("  Extracting embeddings (residuals)...")
    embeddings = model.get_predictions(adata, batch_size=128, get_residuals=True)
    adata.obsm['X_embed'] = embeddings
    
    # Compute UMAP for visualization
    print("  Computing UMAP...")
    sc.pp.neighbors(adata, use_rep='X_embed', n_neighbors=15)
    sc.tl.umap(adata, min_dist=0.3)
    
    print(f"  ✓ Loaded {model_name}: {adata.n_obs} cells")
    
    return adata


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_comparison(adata_baseline, adata_ba, metrics_baseline, metrics_ba, output_dir):
    """Generate comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: UMAP comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Baseline
    sc.pl.umap(adata_baseline, color='cell_type', ax=axes[0, 0], show=False,
               title='Baseline: Cell Types', frameon=False)
    sc.pl.umap(adata_baseline, color='batch', ax=axes[0, 1], show=False,
               title='Baseline: Batches', frameon=False)
    axes[0, 2].scatter(adata_baseline.obs['x_coord'], adata_baseline.obs['y_coord'],
                       c=pd.Categorical(adata_baseline.obs['batch']).codes,
                       s=5, alpha=0.6, cmap='Set1')
    axes[0, 2].set_title('Baseline: Spatial')
    
    # Row 2: BA-AMICI
    sc.pl.umap(adata_ba, color='cell_type', ax=axes[1, 0], show=False,
               title='BA-AMICI: Cell Types', frameon=False)
    sc.pl.umap(adata_ba, color='batch', ax=axes[1, 1], show=False,
               title='BA-AMICI: Batches (Corrected)', frameon=False)
    axes[1, 2].scatter(adata_ba.obs['x_coord'], adata_ba.obs['y_coord'],
                       c=pd.Categorical(adata_ba.obs['batch']).codes,
                       s=5, alpha=0.6, cmap='Set1')
    axes[1, 2].set_title('BA-AMICI: Spatial')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure1_umap_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'figure1_umap_comparison.png'}")
    
    # Figure 2: Metrics bars
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = ['Baseline', 'BA-AMICI']
    x = np.arange(len(models))
    
    # Panel A: Batch mixing
    metrics_data = {
        'iLISI (norm)': [metrics_baseline['iLISI_normalized'], metrics_ba['iLISI_normalized']],
        'kBET': [metrics_baseline['kBET_acceptance'], metrics_ba['kBET_acceptance']],
    }
    width = 0.35
    axes[0].bar(x - width/2, metrics_data['iLISI (norm)'], width, label='iLISI (norm)', alpha=0.8)
    axes[0].bar(x + width/2, metrics_data['kBET'], width, label='kBET', alpha=0.8)
    axes[0].set_ylabel('Score (Higher = Better)')
    axes[0].set_title('Batch Mixing')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].legend()
    axes[0].set_ylim([0, 1.1])
    
    # Panel B: Biology preservation
    bio_data = [metrics_baseline['silhouette_celltype'], metrics_ba['silhouette_celltype']]
    bars = axes[1].bar(models, bio_data, alpha=0.8, color=['#ff7f0e', '#2ca02c'])
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Cell Type Preservation')
    
    # Panel C: Composite
    composite = [metrics_baseline['composite_score'], metrics_ba['composite_score']]
    bars = axes[2].bar(models, composite, alpha=0.8, color=['#ff7f0e', '#2ca02c'])
    axes[2].set_ylabel('Composite Score')
    axes[2].set_title('Overall Performance')
    for bar in bars:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure2_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'figure2_metrics_comparison.png'}")
    
    # Figure 3: LISI distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ilisi_baseline = compute_lisi(adata_baseline.obsm['X_embed'], 
                                   adata_baseline.obs['batch'].values)
    ilisi_ba = compute_lisi(adata_ba.obsm['X_embed'],
                            adata_ba.obs['batch'].values)
    
    data_ilisi = pd.DataFrame({
        'iLISI': np.concatenate([ilisi_baseline, ilisi_ba]),
        'Model': ['Baseline'] * len(ilisi_baseline) + ['BA-AMICI'] * len(ilisi_ba)
    })
    
    sns.violinplot(data=data_ilisi, x='Model', y='iLISI', ax=axes[0])
    n_batches = len(adata_baseline.obs['batch'].unique())
    axes[0].axhline(y=n_batches, color='red', linestyle='--', alpha=0.5, label=f'Ideal ({n_batches})')
    axes[0].set_title('iLISI Distribution (Batch Mixing)')
    axes[0].legend()
    
    clisi_baseline = compute_lisi(adata_baseline.obsm['X_embed'],
                                   adata_baseline.obs['cell_type'].values)
    clisi_ba = compute_lisi(adata_ba.obsm['X_embed'],
                            adata_ba.obs['cell_type'].values)
    
    data_clisi = pd.DataFrame({
        'cLISI': np.concatenate([clisi_baseline, clisi_ba]),
        'Model': ['Baseline'] * len(clisi_baseline) + ['BA-AMICI'] * len(clisi_ba)
    })
    
    sns.violinplot(data=data_clisi, x='Model', y='cLISI', ax=axes[1])
    axes[1].set_title('cLISI Distribution (Cell Type Purity)')
    axes[1].set_ylabel('cLISI (Lower = Purer)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_lisi_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'figure3_lisi_distributions.png'}")


def save_metrics_table(metrics_baseline, metrics_ba, output_dir):
    """Save comparison table."""
    output_dir = Path(output_dir)
    
    df = pd.DataFrame({
        'Metric': list(metrics_baseline.keys()),
        'Baseline': list(metrics_baseline.values()),
        'BA-AMICI': list(metrics_ba.values()),
    })
    
    # Compute improvement
    df['Improvement (%)'] = ((df['BA-AMICI'] - df['Baseline']) / 
                             (np.abs(df['Baseline']) + 1e-10) * 100).round(2)
    
    df['Baseline'] = df['Baseline'].round(4)
    df['BA-AMICI'] = df['BA-AMICI'].round(4)
    
    csv_path = output_dir / 'metrics_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved metrics: {csv_path}")
    
    return df


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # Configuration
    DATA_PATH = PROJECT_ROOT / "pbmc_data" / "ba_amici_benchmark" / "replicate_00.h5ad"
    BASELINE_PATH = SCRIPT_DIR / "results" / "baseline_amici_replicate_00"
    BA_AMICI_PATH = SCRIPT_DIR / "results" / "ba_amici_replicate_00"
    OUTPUT_DIR = SCRIPT_DIR / "validation_results"
    
    print("\n" + "="*70)
    print("BA-AMICI BATCH CORRECTION VALIDATION (FIXED)")
    print("="*70)
    
    # First, diagnose data quality
    print("\nStep 1: Checking data quality...")
    adata_raw = sc.read_h5ad(DATA_PATH)
    if 'spatial' not in adata_raw.obsm:
        adata_raw.obsm['spatial'] = np.column_stack([
            adata_raw.obs['x_coord'].values,
            adata_raw.obs['y_coord'].values
        ])
    data_quality = diagnose_data_quality(adata_raw)
    
    # Load models
    print("\nStep 2: Loading models and extracting embeddings...")
    adata_baseline = load_model_and_extract(
        BASELINE_PATH, DATA_PATH, "Baseline AMICI"
    )
    adata_ba = load_model_and_extract(
        BA_AMICI_PATH, DATA_PATH, "BA-AMICI"
    )
    
    # Compute metrics
    print("\nStep 3: Computing evaluation metrics...")
    metrics_baseline = compute_all_metrics(
        adata_baseline.obsm['X_embed'],
        adata_baseline.obs['batch'].values,
        adata_baseline.obs['cell_type'].values,
        "Baseline AMICI"
    )
    metrics_ba = compute_all_metrics(
        adata_ba.obsm['X_embed'],
        adata_ba.obs['batch'].values,
        adata_ba.obs['cell_type'].values,
        "BA-AMICI"
    )
    
    # Generate figures
    print("\nStep 4: Generating figures...")
    plot_comparison(adata_baseline, adata_ba, metrics_baseline, metrics_ba, OUTPUT_DIR)
    
    # Save metrics table
    df_metrics = save_metrics_table(metrics_baseline, metrics_ba, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    
    print("\nKey Results:")
    print(f"  iLISI:     {metrics_baseline['iLISI_mean']:.3f} → {metrics_ba['iLISI_mean']:.3f} " +
          f"({(metrics_ba['iLISI_normalized'] - metrics_baseline['iLISI_normalized'])*100:+.1f}% normalized)")
    print(f"  kBET:      {metrics_baseline['kBET_acceptance']:.3f} → {metrics_ba['kBET_acceptance']:.3f}")
    print(f"  Celltype:  {metrics_baseline['silhouette_celltype']:.3f} → {metrics_ba['silhouette_celltype']:.3f}")
    print(f"  Composite: {metrics_baseline['composite_score']:.3f} → {metrics_ba['composite_score']:.3f}")
    
    improvement = ((metrics_ba['composite_score'] - metrics_baseline['composite_score']) / 
                   metrics_baseline['composite_score'] * 100)
    print(f"\n  Overall Improvement: {improvement:+.1f}%")
    
    # Interpretation
    print("\n" + "-"*70)
    print("INTERPRETATION:")
    if metrics_ba['iLISI_normalized'] > metrics_baseline['iLISI_normalized'] + 0.1:
        print("  ✓ BA-AMICI significantly improves batch mixing")
    else:
        print("Batch mixing improvement is minimal")
        print("    Possible causes:")
        print("    - Data doesn't have strong enough batch effects")
        print("    - Baseline already handles batches well")
        print("    - Need to tune lambda_adv or architecture")
    
    if metrics_ba['silhouette_celltype'] >= metrics_baseline['silhouette_celltype'] * 0.95:
        print("  ✓ Cell type structure is preserved")
    else:
        print("Some cell type structure may be lost")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()