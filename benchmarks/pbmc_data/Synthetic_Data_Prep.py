"""
BA-AMICI PBMC Dataset Preparation Script
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "pbmc_data"
DATA_DIR.mkdir(exist_ok=True)

FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

import scanpy as sc
sc.settings.figdir = str(FIG_DIR)     # MUST be str, not Path
sc.settings.autoshow = False

print(f"\n{'='*60}")
print("BA-AMICI PBMC Dataset Preparation Pipeline")
print("Using Scanpy's built-in datasets")
print(f"{'='*60}")

def download_pbmc_datasets():
    """
    Download multiple PBMC datasets using Scanpy's built-in functions
    These will serve as different batches with natural variation
    """
    datasets = {}
    
    print("\n" + "="*60)
    print("Downloading PBMC datasets...")
    print("="*60)
    
    # Strategy: Use the FULL pbmc3k dataset and create multiple batches
    # by subsetting and adding technical variation
    
    print("\n1. Loading pbmc3k dataset (full)...")
    try:
        adata_full = sc.datasets.pbmc3k()
        print(f"   ✓ Loaded: {adata_full.n_obs} cells, {adata_full.n_vars} genes")
        
        # Create 3 batches by splitting the data
        n_cells = adata_full.n_obs
        n_per_batch = n_cells // 3
        
        # Batch 1: First third
        adata_b1 = adata_full[:n_per_batch].copy()
        adata_b1.obs['batch'] = 'batch_1'
        adata_b1.obs['batch_id'] = 0
        datasets['batch_1'] = adata_b1
        print(f"   Created batch_1: {adata_b1.n_obs} cells")
        
        # Batch 2: Second third
        adata_b2 = adata_full[n_per_batch:2*n_per_batch].copy()
        adata_b2.obs['batch'] = 'batch_2'
        adata_b2.obs['batch_id'] = 1
        datasets['batch_2'] = adata_b2
        print(f"   Created batch_2: {adata_b2.n_obs} cells")
        
        # Batch 3: Remaining cells
        adata_b3 = adata_full[2*n_per_batch:].copy()
        adata_b3.obs['batch'] = 'batch_3'
        adata_b3.obs['batch_id'] = 2
        datasets['batch_3'] = adata_b3
        print(f"   Created batch_3: {adata_b3.n_obs} cells")
        
        print(f"\n✓ Created 3 batches from pbmc3k dataset")
        print(f"  All batches share the same {adata_full.n_vars} genes")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return download_alternative_datasets()
    
    return datasets

def download_alternative_datasets():
    """
    Alternative: Create artificial batches from single dataset
    by subsampling and adding technical variation
    """
    print("\n" + "="*60)
    print("Using alternative approach: Creating artificial batches")
    print("="*60)
    
    # Load full pbmc3k dataset
    print("\nLoading pbmc3k dataset...")
    adata = sc.datasets.pbmc3k()
    
    # Make 3 copies with different subsampling (simulating different batches)
    datasets = {}
    
    # Batch 1: Random subsample 1
    np.random.seed(42)
    idx1 = np.random.choice(adata.n_obs, size=min(1000, adata.n_obs), replace=False)
    adata_b1 = adata[idx1].copy()
    adata_b1.obs['batch'] = 'batch_1'
    adata_b1.obs['batch_id'] = 0
    datasets['batch_1'] = adata_b1
    print(f"Batch 1: {adata_b1.n_obs} cells")
    
    # Batch 2: Random subsample 2 (different cells)
    np.random.seed(123)
    remaining_idx = np.setdiff1d(np.arange(adata.n_obs), idx1)
    idx2 = np.random.choice(remaining_idx, size=min(1000, len(remaining_idx)), replace=False)
    adata_b2 = adata[idx2].copy()
    adata_b2.obs['batch'] = 'batch_2'
    adata_b2.obs['batch_id'] = 1
    datasets['batch_2'] = adata_b2
    print(f"Batch 2: {adata_b2.n_obs} cells")
    
    # Batch 3: Remaining cells
    idx3 = np.setdiff1d(np.setdiff1d(np.arange(adata.n_obs), idx1), idx2)
    if len(idx3) > 0:
        adata_b3 = adata[idx3[:1000]].copy()
        adata_b3.obs['batch'] = 'batch_3'
        adata_b3.obs['batch_id'] = 2
        datasets['batch_3'] = adata_b3
        print(f"Batch 3: {adata_b3.n_obs} cells")
    
    return datasets

def preprocess_and_combine(datasets):
    """Preprocess individual datasets before combining"""
    print(f"\n{'='*60}")
    print("Preprocessing individual datasets...")
    print(f"{'='*60}")
    
    processed_datasets = []
    
    for name, adata in datasets.items():
        print(f"\nProcessing {name}...")
        
        # Make a copy to avoid modifying original
        adata = adata.copy()
        
        # Basic QC
        sc.pp.calculate_qc_metrics(adata, inplace=True)
        
        # Filter cells and genes
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        
        print(f"  After filtering: {adata.n_obs} cells, {adata.n_vars} genes")
        
        # Store raw counts
        adata.layers['counts'] = adata.X.copy()
        
        # Normalize
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        processed_datasets.append(adata)
    
    # Concatenate - since all batches are from same source, use inner join
    print(f"\n{'='*60}")
    print("Combining datasets...")
    print(f"{'='*60}")
    
    adata_combined = sc.concat(
        processed_datasets, 
        join='inner',  # All have same genes
        label='batch_source',
        keys=[ad.obs['batch'][0] for ad in processed_datasets]
    )
    
    print(f"\nCombined dataset: {adata_combined.n_obs} cells, {adata_combined.n_vars} genes")
    print(f"Batches: {list(adata_combined.obs['batch'].unique())}")
    print("\nBatch distribution:")
    print(adata_combined.obs['batch'].value_counts())
    
    return adata_combined

def select_highly_variable_genes(adata, n_top_genes=500):
    """Select highly variable genes across batches"""
    print(f"\n{'='*60}")
    print(f"Selecting highly variable genes...")
    print(f"{'='*60}")
    
    # Adjust n_top_genes if we have fewer genes
    n_genes = adata.n_vars
    n_top_genes = min(n_top_genes, n_genes)
    
    print(f"Available genes: {n_genes}")
    print(f"Selecting top {n_top_genes} highly variable genes")
    
    # Use seurat flavor which is more robust
    try:
        sc.pp.highly_variable_genes(
            adata, 
            n_top_genes=n_top_genes,
            batch_key='batch',
            flavor='seurat'
        )
    except:
        # Fallback: simple variance-based selection without batch correction
        print("  Using simple variance-based selection...")
        sc.pp.highly_variable_genes(
            adata, 
            n_top_genes=n_top_genes,
            flavor='seurat'
        )
    
    n_hvg = sum(adata.var['highly_variable'])
    print(f"Selected {n_hvg} highly variable genes")
    
    # Keep only HVGs
    adata = adata[:, adata.var['highly_variable']].copy()
    
    return adata

def analyze_batch_effects(adata):
    """Analyze batch effects in the data"""
    print(f"\n{'='*60}")
    print("Analyzing batch effects...")
    print(f"{'='*60}")
    
    # PCA
    print("\nComputing PCA...")
    sc.tl.pca(adata, svd_solver='arpack')
    
    # Neighbors and UMAP
    print("Computing neighbors and UMAP...")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)
    
    # Clustering
    print("Performing Leiden clustering...")
    sc.tl.leiden(adata, resolution=0.5)
    
    # Print batch statistics
    print("\n" + "="*60)
    print("Batch Effect Summary:")
    print("="*60)
    
    for batch in adata.obs['batch'].unique():
        batch_cells = adata[adata.obs['batch'] == batch]
        print(f"\n{batch}:")
        print(f"  Cells: {batch_cells.n_obs}")
        print(f"  Mean genes: {batch_cells.obs['n_genes_by_counts'].mean():.0f}")
        print(f"  Mean counts: {batch_cells.obs['total_counts'].mean():.0f}")
    
    return adata

def save_results(adata, filename="pbmc_multi_batch.h5ad"):
    """Save processed data"""
    output_path = DATA_DIR / filename
    print(f"\n{'='*60}")
    print(f"Saving results to {output_path}...")
    print(f"{'='*60}")
    
    adata.write(output_path)
    print("✓ Save complete!")
    
    return output_path

def visualize_results(adata):
    """Create visualization plots"""

    print(f"\n{'='*60}")
    print("Creating visualizations...")
    print(f"{'='*60}")

    sc.set_figure_params(dpi=100, facecolor='white', frameon=False)

    print("\nGenerating UMAP plots...")
    print("Scanpy saving to:", sc.settings.figdir)

    sc.pl.umap(adata, color=['batch', 'leiden'], 
               save='_batch_overview.png',
               show=False)

    print("Generating PCA plot...")
    sc.pl.pca(adata, color='batch', 
              save='_batch_pca.png',
              show=False)

    print(f"\n✓ Plots saved to {FIG_DIR}/")

def main():
    """Main execution pipeline"""
    
    # Step 1: Download datasets
    datasets = download_pbmc_datasets()
    
    if len(datasets) < 2:
        print("\n✗ Error: Could not load sufficient datasets!")
        return None
    
    print(f"\n✓ Successfully loaded {len(datasets)} datasets")
    
    # Step 2: Preprocess and combine
    adata = preprocess_and_combine(datasets)
    
    # Step 3: Select highly variable genes
    adata = select_highly_variable_genes(adata, n_top_genes=500)
    
    # Step 4: Analyze batch effects
    adata = analyze_batch_effects(adata)
    
    # Step 5: Visualize
    visualize_results(adata)
    
    # Step 6: Save
    output_path = save_results(adata)
    
    # Final summary
    print(f"\n{'='*60}")
    print("Pipeline Complete!")
    print(f"{'='*60}")
    print(f"\n✓ Processed data: {output_path}")
    print(f"✓ Total cells: {adata.n_obs}")
    print(f"✓ Total genes: {adata.n_vars}")
    print(f"✓ Batches: {list(adata.obs['batch'].unique())}")
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Run: python ba_amici_semisynthetic.py")
    print("2. This will generate 10 semi-synthetic replicates")
    print("3. Each replicate will have multiple batches with batch effects")
    print("4. Use these for BA-AMICI training and evaluation")
    
    return adata

if __name__ == "__main__":
    adata = main()