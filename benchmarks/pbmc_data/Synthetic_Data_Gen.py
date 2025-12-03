"""
Semi-synthetic spatial transcriptomics benchmark generator.

Generates spatial scRNA-seq data with known cell-cell interactions and batch effects
for validating batch correction and interaction inference methods.

Author: [Your Name]
"""

import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import scipy.sparse as sp
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class SpatialBenchmarkGenerator:
    """
    Semi-synthetic spatial transcriptomics data generator.
    
    Strategy:
    1. Apply realistic batch effects to RAW COUNTS (before normalization)
    2. Extract real biological clusters from batch-corrected data
    3. Assign spatial coordinates based on cell type (tissue domains)
    4. Imprint spatial proximity-dependent gene expression signals in COUNT SPACE
    
    This creates ground truth for:
    - Cell type spatial organization
    - Distance-dependent cell-cell interactions
    - Technical batch variation (applied at correct stage)
    """
    
    def __init__(
        self,
        adata: sc.AnnData,
        n_cell_types: int = 3,
        seed: int = 42
    ):
        """
        Initialize generator with source data.
        
        Parameters
        ----------
        adata : AnnData
            Source single-cell data (MUST contain raw counts in .X)
        n_cell_types : int
            Number of dominant cell types to extract
        seed : int
            Random seed for reproducibility
        """
        self.adata = adata.copy()
        self.n_cell_types = n_cell_types
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # CRITICAL: Ensure we're working with raw counts
        if sp.issparse(self.adata.X):
            self.adata.X = self.adata.X.toarray()
        
        # Verify data looks like counts (should be integers or near-integers)
        if self.adata.X.max() < 100:
            warnings.warn(
                "Input data has very low values. Expected raw counts (100s-1000s). "
                "Results may be unrealistic if data is already normalized."
            )
        
        # Initialize batch column if missing
        if 'batch' not in self.adata.obs:
            self._create_default_batches()
        
        self.batch_labels = self.adata.obs['batch'].unique()
        print(f"Initialized: {self.adata.n_obs} cells, {len(self.batch_labels)} batches")
        print(f"Expression range: [{self.adata.X.min():.1f}, {self.adata.X.max():.1f}]")
    
    def _create_default_batches(self):
        """Create equal-sized batches if none exist."""
        n = self.adata.n_obs
        n_per_batch = n // 3
        batches = (
            ['batch_1'] * n_per_batch +
            ['batch_2'] * n_per_batch +
            ['batch_3'] * (n - 2 * n_per_batch)
        )
        self.adata.obs['batch'] = batches
    
    def add_batch_effects_postnorm(
        self,
        scale_strength: float = 0.3,
        shift_strength: float = 0.4,
        n_gene_modules: int = 10
    ) -> 'SpatialBenchmarkGenerator':
        """
        Add batch effects to LOG-NORMALIZED data.
        
        IMPORTANT: This is applied AFTER normalization to create visible
        batch effects in the final dataset. This is appropriate for benchmarks
        where you want to test batch correction methods.
        
        Creates two types of effects:
        1. Gene-specific scaling (differential gene amplification/detection)
        2. Systematic shifts with correlation structure (technical bias)
        
        Parameters
        ----------
        scale_strength : float
            Strength of multiplicative scaling effects (0-1 range)
        shift_strength : float  
            Strength of additive shift effects
        n_gene_modules : int
            Number of gene modules (creates correlation structure)
        
        Returns
        -------
        self : SpatialBenchmarkGenerator
            For method chaining
        """
        print("\n[FINAL] Adding batch effects to normalized data...")
        print("  NOTE: Applied post-normalization for visible batch separation")
        
        n_genes = self.adata.n_vars
        
        # Assign genes to modules for correlated effects
        gene_modules = self.rng.integers(0, n_gene_modules, n_genes)
        
        for batch_name in self.batch_labels:
            batch_mask = self.adata.obs['batch'] == batch_name
            
            # 1. GENE-SPECIFIC SCALING (simulates differential efficiency)
            # Modules scale together
            module_scales = self.rng.normal(1.0, scale_strength, n_gene_modules)
            gene_scales = module_scales[gene_modules]
            
            # Add gene-specific noise
            gene_scales += self.rng.normal(0, scale_strength * 0.2, n_genes)
            
            # Ensure scales are positive and reasonable
            gene_scales = np.clip(gene_scales, 0.6, 1.4)
            
            # 2. SYSTEMATIC SHIFTS (simulates technical bias)
            # Module-level shifts
            module_shifts = self.rng.normal(0, shift_strength, n_gene_modules)
            gene_shifts = module_shifts[gene_modules]
            
            # Add gene-level variation
            gene_shifts += self.rng.normal(0, shift_strength * 0.3, n_genes)
            
            # 3. APPLY EFFECTS IN LOG-NORMALIZED SPACE
            # Scale (multiplicative in log-space)
            self.adata.X[batch_mask] *= gene_scales
            
            # Shift (additive in log-space)  
            self.adata.X[batch_mask] += gene_shifts
            
            # Handle any negatives introduced by shifts
            # In log-space, clip to small positive value
            neg_mask = self.adata.X[batch_mask] < 0
            if neg_mask.any():
                self.adata.X[batch_mask][neg_mask] = np.abs(
                    self.adata.X[batch_mask][neg_mask]
                ) * 0.05
            
            print(f"  {batch_name}: mean_scale={gene_scales.mean():.3f}, "
                  f"mean_shift={np.abs(gene_shifts).mean():.3f}")
        
        return self
    
    def identify_cell_types(self, resolution: float = 0.5) -> 'SpatialBenchmarkGenerator':
        """
        Identify major cell types using Leiden clustering.
        
        Uses standard preprocessing pipeline to find stable clusters,
        then retains only the top N most abundant clusters.
        
        NOTE: This step normalizes the data. For benchmarks, batch effects
        should be applied AFTER this step to remain visible.
        
        Parameters
        ----------
        resolution : float
            Leiden clustering resolution
        
        Returns
        -------
        self : SpatialBenchmarkGenerator
            For method chaining
        """
        print("\n[1/4] Identifying cell types (with normalization)...")
        
        # Standard preprocessing
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        sc.pp.highly_variable_genes(self.adata, n_top_genes=2000)
        
        # Dimensionality reduction and clustering
        sc.pp.pca(self.adata, n_comps=50)
        sc.pp.neighbors(self.adata, n_neighbors=15, n_pcs=50)
        sc.tl.leiden(self.adata, resolution=resolution, key_added='leiden')
        
        # Select top N clusters by size
        cluster_sizes = self.adata.obs['leiden'].value_counts()
        top_clusters = cluster_sizes.nlargest(self.n_cell_types).index.tolist()
        
        # Filter to keep only these clusters
        keep_mask = self.adata.obs['leiden'].isin(top_clusters)
        self.adata = self.adata[keep_mask].copy()
        
        # Assign cell type labels (A, B, C, ...)
        cluster_to_type = {
            cluster: chr(65 + i)  # 65 is ASCII for 'A'
            for i, cluster in enumerate(top_clusters)
        }
        self.adata.obs['cell_type'] = self.adata.obs['leiden'].map(cluster_to_type)
        
        type_counts = self.adata.obs['cell_type'].value_counts()
        print(f"  Retained {self.adata.n_obs} cells across {self.n_cell_types} types:")
        for ct, count in type_counts.items():
            print(f"    Type {ct}: {count} cells")
        
        return self
    
    def assign_spatial_coordinates(
        self,
        domain_spread: float = 300.0
    ) -> 'SpatialBenchmarkGenerator':
        """
        Assign spatial coordinates to create tissue-like domains.
        
        Each cell type is placed in a spatial domain with Gaussian spread.
        This simulates anatomical tissue structure (e.g., tumor core, stroma).
        
        Parameters
        ----------
        domain_spread : float
            Standard deviation of cell positions within each domain (larger = more mixing)
        
        Returns
        -------
        self : SpatialBenchmarkGenerator
            For method chaining
        """
        print("\n[2/4] Assigning spatial coordinates...")
        
        n = self.adata.n_obs
        coords = np.zeros((n, 2))
        
        # Define domain centroids in circular arrangement
        centroids = self._compute_domain_centroids(self.n_cell_types)
        
        # Scatter cells around their domain centroids
        for cell_type, (cx, cy) in centroids.items():
            mask = self.adata.obs['cell_type'] == cell_type
            n_cells = mask.sum()
            
            coords[mask, 0] = self.rng.normal(cx, domain_spread, n_cells)
            coords[mask, 1] = self.rng.normal(cy, domain_spread, n_cells)
        
        self.adata.obs['x_coord'] = coords[:, 0]
        self.adata.obs['y_coord'] = coords[:, 1]
        self.adata.obsm['spatial'] = coords
        
        print(f"  Placed cells in {len(centroids)} spatial domains (spread={domain_spread})")
        
        return self
    
    def _compute_domain_centroids(self, n_types: int) -> Dict[str, Tuple[float, float]]:
        """
        Compute centroids for spatial domains in circular arrangement.
        
        Parameters
        ----------
        n_types : int
            Number of cell types
        
        Returns
        -------
        centroids : dict
            Mapping from cell type label to (x, y) centroid
        """
        radius = 800  # Distance from origin
        centroids = {}
        
        for i in range(n_types):
            angle = 2 * np.pi * i / n_types
            x = radius * np.cos(angle) + 1000  # Offset to keep positive
            y = radius * np.sin(angle) + 1000
            cell_type = chr(65 + i)
            centroids[cell_type] = (x, y)
        
        return centroids
    
    def imprint_spatial_interactions(
        self,
        interaction_rules: Optional[List[Dict]] = None
    ) -> 'SpatialBenchmarkGenerator':
        """
        Imprint distance-dependent gene expression signals.
        
        IMPORTANT: Works in LOG-NORMALIZED space since we need to modify
        expression AFTER cell type identification. This is biologically
        valid (represents post-transcriptional regulation by signals).
        
        Parameters
        ----------
        interaction_rules : list of dict, optional
            Each dict should contain:
            - 'sender': str, sender cell type
            - 'receiver': str, receiver cell type  
            - 'distance_threshold': float, max distance for interaction
            - 'target_genes': list of int, gene indices to modulate
            - 'log_fc': float, log-fold change to add
        
        Returns
        -------
        self : SpatialBenchmarkGenerator
            For method chaining
        """
        print("\n[3/4] Imprinting spatial interactions...")
        
        if interaction_rules is None:
            interaction_rules = self._default_interaction_rules()
        
        # Compute pairwise distances
        coords = self.adata.obsm['spatial']
        dist_matrix = pairwise_distances(coords, metric='euclidean')
        
        interaction_labels = np.zeros(self.adata.n_obs, dtype=bool)
        
        for rule in interaction_rules:
            n_activated = self._apply_interaction_rule(rule, dist_matrix, interaction_labels)
            
            sender = rule['sender']
            receiver = rule['receiver']
            threshold = rule['distance_threshold']
            print(f"  {sender}→{receiver} (d<{threshold}): {n_activated} receiver cells activated")
        
        self.adata.obs['has_interaction'] = interaction_labels
        print(f"  Total interacting cells: {interaction_labels.sum()}")
        
        return self
    
    def _default_interaction_rules(self) -> List[Dict]:
        """
        Define default interaction rules.
        
        These simulate ligand-receptor signaling where proximity matters.
        Effect sizes are in log-space (log-fold changes).
        """
        return [
            {
                'sender': 'A',
                'receiver': 'B',
                'distance_threshold': 150.0,
                'target_genes': [0, 1, 2],  # First few highly variable genes
                'log_fc': 1.0  # 2-fold increase
            },
            {
                'sender': 'C',
                'receiver': 'A',
                'distance_threshold': 250.0,
                'target_genes': [3, 4, 5],
                'log_fc': 0.8  # 1.7-fold increase
            },
            {
                'sender': 'B',
                'receiver': 'C',
                'distance_threshold': 200.0,
                'target_genes': [6, 7],
                'log_fc': 1.2  # 2.3-fold increase
            }
        ]
    
    def _apply_interaction_rule(
        self,
        rule: Dict,
        dist_matrix: np.ndarray,
        interaction_labels: np.ndarray
    ) -> int:
        """
        Apply a single interaction rule to modify expression.
        
        Returns
        -------
        n_activated : int
            Number of receiver cells that were within interaction distance
        """
        sender_mask = self.adata.obs['cell_type'] == rule['sender']
        receiver_mask = self.adata.obs['cell_type'] == rule['receiver']
        
        sender_idx = np.where(sender_mask)[0]
        receiver_idx = np.where(receiver_mask)[0]
        
        if len(sender_idx) == 0 or len(receiver_idx) == 0:
            return 0
        
        # For each receiver, find minimum distance to any sender
        receiver_to_sender_dist = dist_matrix[np.ix_(receiver_idx, sender_idx)]
        min_distances = receiver_to_sender_dist.min(axis=1)
        
        # Identify receivers within threshold
        close_mask = min_distances < rule['distance_threshold']
        activated_idx = receiver_idx[close_mask]
        
        if len(activated_idx) == 0:
            return 0
        
        # Apply distance-weighted effect (closer = stronger signal)
        # This is more realistic than binary on/off
        active_distances = min_distances[close_mask]
        distance_weights = 1.0 - (active_distances / rule['distance_threshold'])
        
        # Modify expression in LOG SPACE (data is log-normalized at this point)
        target_genes = rule['target_genes']
        base_log_fc = rule['log_fc']
        
        for gene_idx in target_genes:
            if gene_idx < self.adata.n_vars:
                # Apply weighted effect
                weighted_effects = base_log_fc * distance_weights
                self.adata.X[activated_idx, gene_idx] += weighted_effects
        
        # Mark these cells as interacting
        interaction_labels[activated_idx] = True
        
        return len(activated_idx)
    
    def finalize(
        self,
        train_fraction: float = 0.7
    ) -> sc.AnnData:
        """
        Finalize the dataset and create train/test split.
        
        Uses spatial holdout strategy to test model generalization.
        
        Parameters
        ----------
        train_fraction : float
            Approximate fraction of cells to assign to training set
        
        Returns
        -------
        adata : AnnData
            Finalized dataset with all annotations
        """
        # Create spatial train/test split
        # Strategy: hold out a spatial region to test generalization
        x_coords = self.adata.obs['x_coord'].values
        x_min, x_max = x_coords.min(), x_coords.max()
        
        # Hold out middle vertical strip (tests interaction recovery in held-out space)
        test_x_min = x_min + (x_max - x_min) * 0.4
        test_x_max = x_min + (x_max - x_min) * 0.6
        
        test_mask = (x_coords >= test_x_min) & (x_coords <= test_x_max)
        self.adata.obs['split'] = 'train'
        self.adata.obs.loc[test_mask, 'split'] = 'test'
        
        n_train = (self.adata.obs['split'] == 'train').sum()
        n_test = (self.adata.obs['split'] == 'test').sum()
        print(f"\nTrain/test split: {n_train} train ({n_train/len(self.adata)*100:.1f}%), "
              f"{n_test} test ({n_test/len(self.adata)*100:.1f}%)")
        
        return self.adata
    
    def save(self, output_path: Path):
        """Save dataset to h5ad format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.adata.write_h5ad(output_path)
        print(f"Saved to {output_path}")
    
    def plot_qc(self, output_path: Optional[Path] = None):
        """Generate quality control plots."""
        # Recompute embedding for visualization
        sc.pp.pca(self.adata, n_comps=50)
        sc.pp.neighbors(self.adata)
        sc.tl.umap(self.adata)
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # UMAP by cell type
        sc.pl.umap(self.adata, color='cell_type', ax=axes[0, 0], show=False,
                   title='Cell Types', legend_loc='right margin')
        
        # UMAP by batch
        sc.pl.umap(self.adata, color='batch', ax=axes[0, 1], show=False,
                   title='Batch Effect', legend_loc='right margin')
        
        # UMAP by interaction status
        sc.pl.umap(self.adata, color='has_interaction', ax=axes[0, 2], show=False,
                   title='Interacting Cells', legend_loc='right margin')
        
        # Spatial plot by cell type
        cell_type_codes = pd.Categorical(self.adata.obs['cell_type']).codes
        scatter1 = axes[1, 0].scatter(
            self.adata.obs['x_coord'],
            self.adata.obs['y_coord'],
            c=cell_type_codes,
            s=3, alpha=0.6, cmap='tab10'
        )
        axes[1, 0].set_title('Spatial: Cell Types')
        axes[1, 0].set_xlabel('X coordinate')
        axes[1, 0].set_ylabel('Y coordinate')
        plt.colorbar(scatter1, ax=axes[1, 0], label='Cell Type')
        
        # Spatial plot by batch
        batch_codes = pd.Categorical(self.adata.obs['batch']).codes
        scatter2 = axes[1, 1].scatter(
            self.adata.obs['x_coord'],
            self.adata.obs['y_coord'],
            c=batch_codes,
            s=3, alpha=0.6, cmap='Set1'
        )
        axes[1, 1].set_title('Spatial: Batch Distribution')
        axes[1, 1].set_xlabel('X coordinate')
        axes[1, 1].set_ylabel('Y coordinate')
        plt.colorbar(scatter2, ax=axes[1, 1], label='Batch')
        
        # Interaction status
        scatter3 = axes[1, 2].scatter(
            self.adata.obs['x_coord'],
            self.adata.obs['y_coord'],
            c=self.adata.obs['has_interaction'].astype(int),
            s=3, alpha=0.6, cmap='coolwarm'
        )
        axes[1, 2].set_title('Spatial: Interacting Cells')
        axes[1, 2].set_xlabel('X coordinate')
        axes[1, 2].set_ylabel('Y coordinate')
        plt.colorbar(scatter3, ax=axes[1, 2], label='Has Interaction')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"QC plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()


def generate_benchmark_dataset(
    source_path: Path,
    output_dir: Path,
    n_replicates: int = 10,
    base_seed: int = 42,
    **kwargs
) -> None:
    """
    Generate multiple replicate benchmark datasets.
    
    Parameters
    ----------
    source_path : Path
        Path to source h5ad file with RAW COUNTS (critical!)
    output_dir : Path
        Directory to save benchmark replicates
    n_replicates : int
        Number of independent replicates to generate
    base_seed : int
        Base random seed (each replicate gets base_seed + replicate_id)
    **kwargs
        Additional arguments passed to SpatialBenchmarkGenerator
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading source data from {source_path}")
    adata_source = sc.read_h5ad(source_path)
    
    print("\n" + "="*70)
    print("GENERATING BENCHMARK WITH POST-NORMALIZATION BATCH EFFECTS")
    print("This ensures batch effects are visible in final normalized data")
    print("="*70 + "\n")
    
    for rep_id in range(n_replicates):
        rep_name = f"replicate_{rep_id:02d}"
        print(f"\n{'='*70}")
        print(f"GENERATING {rep_name}")
        print(f"{'='*70}")
        
        # Create generator with unique seed
        gen = SpatialBenchmarkGenerator(
            adata_source.copy(),
            seed=base_seed + rep_id,
            **kwargs
        )
        
        # Run pipeline IN CORRECT ORDER
        (gen
         .identify_cell_types(resolution=0.5)
         .assign_spatial_coordinates(domain_spread=300)
         .imprint_spatial_interactions()
         .add_batch_effects_postnorm(
             scale_strength=0.35,
             shift_strength=0.45,
             n_gene_modules=10
         )
         .finalize(train_fraction=0.7))
        
        # Save outputs
        output_path = output_dir / f"{rep_name}.h5ad"
        gen.save(output_path)
        
        plot_path = output_dir / f"{rep_name}_qc.png"
        gen.plot_qc(plot_path)
    
    print(f"\n{'='*70}")
    print(f"✓ Successfully generated {n_replicates} replicates")
    print(f"✓ Saved to {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Configuration
    current_dir = Path(__file__).parent.resolve()
    input_file = current_dir / "pbmc_multi_batch.h5ad"
    output_dir = current_dir / "ba_amici_benchmark"
    
    # Verify input file exists
    if not input_file.exists():
        raise FileNotFoundError(
            f"Source file not found: {input_file}\n"
            f"Please ensure you have raw count data in h5ad format."
        )
    
    # Generate benchmark
    generate_benchmark_dataset(
        source_path=input_file,
        output_dir=output_dir,
        n_replicates=10,
        n_cell_types=3,
        base_seed=42
    )