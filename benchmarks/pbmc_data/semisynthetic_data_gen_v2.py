"""
BA-AMICI Semi-Synthetic Benchmark Generator v2

DESIGN PHILOSOPHY:
==================
The goal is to create data where:
1. SAME biological interactions exist across ALL batches
2. DIFFERENT technical/batch effects make batches SEPARABLE
3. Batch effects are CONFOUNDED with spatial structure
4. A good batch-aware model should:
   - Mix batches in latent space (high iLISI)
   - Preserve cell type identity (low cLISI) 
   - Recover SAME interactions across batches

KEY DIFFERENCES FROM v1:
========================
- Batches have DIFFERENT cell type proportions (confounding)
- Batches have DIFFERENT spatial organizations (realistic)
- Batch effects applied to RAW COUNTS (biologically correct)
- Same ground-truth interactions across all batches (testable)
- Proper confounding between batch and biology

Author: BA-AMICI Project
"""

import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
import warnings

warnings.filterwarnings("ignore")


class ProperBatchBenchmarkGenerator:
    """
    Generate semi-synthetic spatial transcriptomics data with:
    - Multiple batches with realistic technical variation
    - Confounded batch-biology relationships
    - Known ground-truth cell-cell interactions
    - Proper evaluation metrics
    """
    
    def __init__(
        self,
        source_adata: sc.AnnData,
        n_batches: int = 3,
        n_cell_types: int = 3,
        seed: int = 42
    ):
        """
        Initialize with source single-cell data.
        
        Parameters
        ----------
        source_adata : AnnData
            Source data - should have raw counts or be normalizable
        n_batches : int
            Number of batches to simulate
        n_cell_types : int
            Number of cell types to use
        seed : int
            Random seed
        """
        self.source = source_adata.copy()
        self.n_batches = n_batches
        self.n_cell_types = n_cell_types
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Will hold the generated data
        self.adata = None
        
        # Ground truth storage
        self.interaction_rules = []
        self.batch_effects_applied = {}
        
        print(f"Initialized generator:")
        print(f"  Source: {self.source.n_obs} cells, {self.source.n_vars} genes")
        print(f"  Target: {n_batches} batches, {n_cell_types} cell types")
    
    def prepare_source_data(self, resolution: float = 0.5) -> 'ProperBatchBenchmarkGenerator':
        """
        Prepare source data by identifying cell types.
        
        This step clusters the source data to identify distinct populations
        that we'll use as our cell types.
        """
        print("\n[1/6] Preparing source data...")
        
        # Store raw counts if available
        if 'counts' in self.source.layers:
            counts = self.source.layers['counts']
        else:
            # Assume .X contains counts or normalized data
            counts = self.source.X.copy()
        
        if sp.issparse(counts):
            counts = counts.toarray()
        
        self.source.layers['counts'] = counts
        
        # Normalize for clustering
        sc.pp.normalize_total(self.source, target_sum=1e4)
        sc.pp.log1p(self.source)
        
        # Find highly variable genes
        sc.pp.highly_variable_genes(self.source, n_top_genes=2000, flavor='seurat')
        
        # PCA and clustering
        sc.pp.pca(self.source, n_comps=50, use_highly_variable=True)
        sc.pp.neighbors(self.source, n_neighbors=15, n_pcs=30)
        sc.tl.leiden(self.source, resolution=resolution, key_added='leiden')
        
        # Keep top N clusters by size
        cluster_sizes = self.source.obs['leiden'].value_counts()
        top_clusters = cluster_sizes.nlargest(self.n_cell_types).index.tolist()
        
        # Filter and assign cell type labels
        mask = self.source.obs['leiden'].isin(top_clusters)
        self.source = self.source[mask].copy()
        
        cluster_to_type = {c: chr(65 + i) for i, c in enumerate(top_clusters)}
        self.source.obs['cell_type'] = self.source.obs['leiden'].map(cluster_to_type)
        
        print(f"  Identified {self.n_cell_types} cell types:")
        for ct in sorted(self.source.obs['cell_type'].unique()):
            n = (self.source.obs['cell_type'] == ct).sum()
            print(f"    Type {ct}: {n} cells")
        
        return self
    
    def generate_batches_with_confounding(
        self,
        cells_per_batch: int = 2000,
        proportion_variation: float = 0.3
    ) -> 'ProperBatchBenchmarkGenerator':
        """
        Generate multiple batches with DIFFERENT cell type proportions.
        
        This creates confounding between batch and cell type - a key challenge
        for batch correction methods.
        
        Parameters
        ----------
        cells_per_batch : int
            Approximate number of cells per batch
        proportion_variation : float
            How much cell type proportions vary between batches (0-1)
            Higher = more confounding
        """
        print("\n[2/6] Generating batches with confounded cell type proportions...")
        
        cell_types = sorted(self.source.obs['cell_type'].unique())
        n_types = len(cell_types)
        
        # Generate different proportions for each batch
        # Start with uniform, then perturb
        base_prop = np.ones(n_types) / n_types
        
        batch_proportions = {}
        for b in range(self.n_batches):
            # Create biased proportions - each batch enriched for different type
            props = base_prop.copy()
            
            # Enrich one cell type per batch (rotating)
            enriched_type_idx = b % n_types
            props[enriched_type_idx] += proportion_variation
            
            # Deplete another type
            depleted_type_idx = (b + 1) % n_types
            props[depleted_type_idx] -= proportion_variation * 0.5
            
            # Add random noise
            props += self.rng.normal(0, 0.05, n_types)
            
            # Normalize to sum to 1
            props = np.clip(props, 0.1, 0.6)  # Ensure no type is too rare
            props = props / props.sum()
            
            batch_proportions[f'batch_{b+1}'] = dict(zip(cell_types, props))
        
        # Print batch compositions
        print("  Batch compositions (cell type proportions):")
        for batch_name, props in batch_proportions.items():
            prop_str = ", ".join([f"{ct}:{p:.2f}" for ct, p in props.items()])
            print(f"    {batch_name}: {prop_str}")
        
        # Sample cells for each batch according to proportions
        all_batch_data = []
        
        for batch_name, proportions in batch_proportions.items():
            batch_cells = []
            
            for cell_type, prop in proportions.items():
                n_cells_this_type = int(cells_per_batch * prop)
                
                # Get cells of this type from source
                type_mask = self.source.obs['cell_type'] == cell_type
                available_cells = self.source[type_mask]
                
                if len(available_cells) < n_cells_this_type:
                    # Sample with replacement if needed
                    idx = self.rng.choice(
                        len(available_cells), 
                        size=n_cells_this_type, 
                        replace=True
                    )
                else:
                    idx = self.rng.choice(
                        len(available_cells), 
                        size=n_cells_this_type, 
                        replace=False
                    )
                
                sampled = available_cells[idx].copy()
                sampled.obs['batch'] = batch_name
                batch_cells.append(sampled)
            
            batch_adata = sc.concat(batch_cells, join='outer')
            all_batch_data.append(batch_adata)
        
        # Combine all batches
        self.adata = sc.concat(all_batch_data, join='outer')
        
        # Reset index
        self.adata.obs_names = [f"cell_{i}" for i in range(self.adata.n_obs)]
        
        # Use raw counts from source
        if 'counts' in self.source.layers:
            # Need to properly transfer counts
            self.adata.layers['counts'] = self.adata.X.copy()
        
        print(f"\n  Generated {self.adata.n_obs} total cells across {self.n_batches} batches")
        print("  Batch sizes:")
        for b in self.adata.obs['batch'].unique():
            n = (self.adata.obs['batch'] == b).sum()
            print(f"    {b}: {n} cells")
        
        self.batch_proportions = batch_proportions
        return self
    
    def apply_batch_effects_to_counts(
        self,
        library_size_variation: float = 0.4,
        gene_detection_variation: float = 0.3,
        dropout_variation: float = 0.2
    ) -> 'ProperBatchBenchmarkGenerator':
        """
        Apply realistic batch effects to RAW COUNTS.
        
        This is the CORRECT way to add batch effects - before normalization.
        
        Effects modeled:
        1. Library size differences (sequencing depth)
        2. Gene-specific detection efficiency
        3. Dropout rate variation
        
        Parameters
        ----------
        library_size_variation : float
            How much library sizes vary between batches (coefficient of variation)
        gene_detection_variation : float  
            How much gene detection efficiency varies
        dropout_variation : float
            How much dropout rates vary
        """
        print("\n[3/6] Applying batch effects to raw counts...")
        
        # Work with counts
        if 'counts' in self.adata.layers:
            X = self.adata.layers['counts'].copy()
        else:
            X = self.adata.X.copy()
        
        if sp.issparse(X):
            X = X.toarray()
        
        X = X.astype(np.float64)
        n_genes = X.shape[1]
        
        batch_effects = {}
        
        for batch_name in self.adata.obs['batch'].unique():
            batch_mask = (self.adata.obs['batch'] == batch_name).values
            batch_idx = np.where(batch_mask)[0]
            
            effects = {}
            
            # 1. LIBRARY SIZE EFFECT
            # Different batches have different sequencing depths
            lib_size_factor = self.rng.lognormal(0, library_size_variation)
            lib_size_factor = np.clip(lib_size_factor, 0.5, 2.0)
            effects['library_size_factor'] = lib_size_factor
            
            # 2. GENE-SPECIFIC DETECTION EFFICIENCY
            # Some genes are better/worse detected in different batches
            # This is the key batch effect that needs correction
            gene_factors = self.rng.lognormal(0, gene_detection_variation, n_genes)
            gene_factors = np.clip(gene_factors, 0.3, 3.0)
            effects['gene_factors'] = gene_factors
            
            # 3. DROPOUT VARIATION
            # Different batches have different dropout rates
            dropout_rate = 0.1 + self.rng.uniform(-dropout_variation, dropout_variation)
            dropout_rate = np.clip(dropout_rate, 0.05, 0.4)
            effects['dropout_rate'] = dropout_rate
            
            # Apply effects to this batch
            X_batch = X[batch_idx].copy()
            
            # Apply library size scaling
            X_batch = X_batch * lib_size_factor
            
            # Apply gene-specific factors
            X_batch = X_batch * gene_factors[np.newaxis, :]
            
            # Apply dropout (zero out some counts)
            dropout_mask = self.rng.random(X_batch.shape) < dropout_rate
            X_batch[dropout_mask] = 0
            
            # Ensure counts are non-negative integers
            X_batch = np.maximum(X_batch, 0)
            X_batch = np.round(X_batch).astype(np.float32)
            
            X[batch_idx] = X_batch
            batch_effects[batch_name] = effects
            
            print(f"  {batch_name}: lib_size={lib_size_factor:.2f}, "
                  f"dropout={dropout_rate:.2f}, "
                  f"gene_factor_range=[{gene_factors.min():.2f}, {gene_factors.max():.2f}]")
        
        # Store modified counts
        self.adata.layers['counts_with_batch_effects'] = X
        self.adata.X = X  # Use batch-affected counts as main matrix
        
        self.batch_effects_applied = batch_effects
        
        return self
    
    def assign_spatial_coordinates_per_batch(
        self,
        spatial_patterns: Optional[Dict] = None,
        domain_spread: float = 200.0
    ) -> 'ProperBatchBenchmarkGenerator':
        """
        Assign DIFFERENT spatial organizations to each batch.
        
        This simulates reality where different tissue sections have
        different spatial arrangements of the same cell types.
        
        Parameters
        ----------
        spatial_patterns : dict, optional
            Mapping of batch name to spatial pattern type
        domain_spread : float
            Spread of cells around domain centroids
        """
        print("\n[4/6] Assigning batch-specific spatial coordinates...")
        
        if spatial_patterns is None:
            # Default: different patterns for each batch
            patterns = ['circular', 'striped', 'nested']
            spatial_patterns = {
                f'batch_{i+1}': patterns[i % len(patterns)]
                for i in range(self.n_batches)
            }
        
        coords = np.zeros((self.adata.n_obs, 2))
        
        for batch_name, pattern in spatial_patterns.items():
            batch_mask = (self.adata.obs['batch'] == batch_name).values
            batch_idx = np.where(batch_mask)[0]
            n_cells = len(batch_idx)
            
            cell_types = self.adata.obs.loc[batch_mask, 'cell_type'].values
            unique_types = sorted(set(cell_types))
            
            # Generate coordinates based on pattern
            batch_coords = self._generate_spatial_pattern(
                pattern, cell_types, unique_types, n_cells, domain_spread
            )
            
            # Add batch-specific offset to prevent spatial overlap
            batch_offset = list(spatial_patterns.keys()).index(batch_name)
            batch_coords[:, 0] += batch_offset * 3000  # Separate batches spatially
            
            coords[batch_idx] = batch_coords
            
            print(f"  {batch_name}: {pattern} pattern, {n_cells} cells")
        
        self.adata.obs['x_coord'] = coords[:, 0]
        self.adata.obs['y_coord'] = coords[:, 1]
        self.adata.obsm['spatial'] = coords
        
        return self
    
    def _generate_spatial_pattern(
        self,
        pattern: str,
        cell_types: np.ndarray,
        unique_types: List[str],
        n_cells: int,
        spread: float
    ) -> np.ndarray:
        """Generate coordinates for a specific spatial pattern."""
        
        coords = np.zeros((n_cells, 2))
        n_types = len(unique_types)
        
        if pattern == 'circular':
            # Cell types in circular domains
            radius = 800
            for i, ct in enumerate(unique_types):
                mask = cell_types == ct
                n_ct = mask.sum()
                angle = 2 * np.pi * i / n_types
                cx = radius * np.cos(angle) + 1000
                cy = radius * np.sin(angle) + 1000
                coords[mask, 0] = self.rng.normal(cx, spread, n_ct)
                coords[mask, 1] = self.rng.normal(cy, spread, n_ct)
                
        elif pattern == 'striped':
            # Cell types in horizontal stripes
            stripe_height = 600
            for i, ct in enumerate(unique_types):
                mask = cell_types == ct
                n_ct = mask.sum()
                cy = i * stripe_height + stripe_height/2
                coords[mask, 0] = self.rng.uniform(0, 2000, n_ct)
                coords[mask, 1] = self.rng.normal(cy, spread/2, n_ct)
                
        elif pattern == 'nested':
            # Cell types in nested rings
            for i, ct in enumerate(unique_types):
                mask = cell_types == ct
                n_ct = mask.sum()
                inner_r = i * 300 + 100
                outer_r = inner_r + 250
                # Sample from ring
                r = self.rng.uniform(inner_r, outer_r, n_ct)
                theta = self.rng.uniform(0, 2*np.pi, n_ct)
                coords[mask, 0] = r * np.cos(theta) + 1000
                coords[mask, 1] = r * np.sin(theta) + 1000
        
        else:
            # Default: random scatter
            coords[:, 0] = self.rng.uniform(0, 2000, n_cells)
            coords[:, 1] = self.rng.uniform(0, 2000, n_cells)
        
        return coords
    
    def imprint_consistent_interactions(
        self,
        interaction_rules: Optional[List[Dict]] = None
    ) -> 'ProperBatchBenchmarkGenerator':
        """
        Imprint the SAME cell-cell interactions across ALL batches.
        
        This is the key ground truth - the interactions should be
        identical regardless of batch. A good batch-aware model
        should recover these same interactions in all batches.
        
        Parameters
        ----------
        interaction_rules : list of dict
            Each dict contains:
            - sender: sender cell type
            - receiver: receiver cell type
            - distance_threshold: max interaction distance
            - target_genes: list of gene indices to affect
            - effect_size: fold-change effect (in count space)
        """
        print("\n[5/6] Imprinting consistent interactions across all batches...")
        
        if interaction_rules is None:
            interaction_rules = self._default_interaction_rules()
        
        self.interaction_rules = interaction_rules
        
        # Initialize interaction tracking
        self.adata.obs['is_interacting'] = False
        self.adata.obs['interaction_type'] = 'none'
        
        # Work with counts
        X = self.adata.X.copy()
        if sp.issparse(X):
            X = X.toarray()
        
        # Apply interactions within each batch separately
        # (cells only interact within their own spatial context)
        for batch_name in self.adata.obs['batch'].unique():
            batch_mask = (self.adata.obs['batch'] == batch_name).values
            batch_idx = np.where(batch_mask)[0]
            
            # Get spatial coordinates for this batch
            coords = self.adata.obsm['spatial'][batch_idx]
            
            # Compute pairwise distances within batch
            dist_matrix = pairwise_distances(coords, metric='euclidean')
            
            # Apply each interaction rule
            for rule in interaction_rules:
                n_activated = self._apply_interaction_to_batch(
                    rule, batch_idx, dist_matrix, X
                )
                if n_activated > 0:
                    print(f"    {batch_name}: {rule['sender']}→{rule['receiver']} "
                          f"activated {n_activated} cells")
        
        # Store modified expression
        self.adata.X = X
        
        # Summarize
        n_interacting = self.adata.obs['is_interacting'].sum()
        print(f"\n  Total interacting cells: {n_interacting} "
              f"({n_interacting/self.adata.n_obs*100:.1f}%)")
        
        return self
    
    def _default_interaction_rules(self) -> List[Dict]:
        """Default interaction rules - same as original AMICI paper."""
        n_genes = self.adata.n_vars
        
        return [
            {
                'sender': 'A',
                'receiver': 'B', 
                'distance_threshold': 150.0,
                'target_genes': list(range(min(3, n_genes))),
                'effect_size': 2.0,  # 2-fold increase
                'name': 'A_to_B'
            },
            {
                'sender': 'C',
                'receiver': 'A',
                'distance_threshold': 200.0,
                'target_genes': list(range(3, min(6, n_genes))),
                'effect_size': 1.8,  # 1.8-fold increase
                'name': 'C_to_A'
            },
            {
                'sender': 'B',
                'receiver': 'C',
                'distance_threshold': 180.0,
                'target_genes': list(range(6, min(8, n_genes))),
                'effect_size': 2.2,  # 2.2-fold increase
                'name': 'B_to_C'
            }
        ]
    
    def _apply_interaction_to_batch(
        self,
        rule: Dict,
        batch_idx: np.ndarray,
        dist_matrix: np.ndarray,
        X: np.ndarray
    ) -> int:
        """Apply a single interaction rule within a batch."""
        
        # Get cell types for this batch
        batch_cell_types = self.adata.obs.iloc[batch_idx]['cell_type'].values
        
        # Find senders and receivers within this batch
        sender_local_idx = np.where(batch_cell_types == rule['sender'])[0]
        receiver_local_idx = np.where(batch_cell_types == rule['receiver'])[0]
        
        if len(sender_local_idx) == 0 or len(receiver_local_idx) == 0:
            return 0
        
        # Find receivers close to senders
        receiver_to_sender_dist = dist_matrix[np.ix_(receiver_local_idx, sender_local_idx)]
        min_distances = receiver_to_sender_dist.min(axis=1)
        
        # Receivers within threshold
        close_mask = min_distances < rule['distance_threshold']
        activated_local_idx = receiver_local_idx[close_mask]
        
        if len(activated_local_idx) == 0:
            return 0
        
        # Convert to global indices
        activated_global_idx = batch_idx[activated_local_idx]
        
        # Distance-weighted effect (closer = stronger)
        active_distances = min_distances[close_mask]
        distance_weights = 1.0 - (active_distances / rule['distance_threshold'])
        
        # Apply effect to target genes (multiplicative in count space)
        for gene_idx in rule['target_genes']:
            if gene_idx < X.shape[1]:
                # Weighted fold-change effect
                fold_change = 1.0 + (rule['effect_size'] - 1.0) * distance_weights
                X[activated_global_idx, gene_idx] *= fold_change
        
        # Mark cells as interacting
        self.adata.obs.iloc[activated_global_idx, 
                           self.adata.obs.columns.get_loc('is_interacting')] = True
        self.adata.obs.iloc[activated_global_idx,
                           self.adata.obs.columns.get_loc('interaction_type')] = rule['name']
        
        return len(activated_global_idx)
    
    def normalize_and_finalize(self) -> sc.AnnData:
        """
        Normalize data and create final dataset with all annotations.
        """
        print("\n[6/6] Normalizing and finalizing...")
        
        # Store raw counts before normalization
        self.adata.layers['counts_final'] = self.adata.X.copy()
        
        # Standard normalization
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        
        # Store normalized values
        self.adata.layers['normalized'] = self.adata.X.copy()
        
        # Compute embeddings for visualization
        sc.pp.highly_variable_genes(self.adata, n_top_genes=500, flavor='seurat')
        sc.pp.pca(self.adata, n_comps=50, use_highly_variable=True)
        sc.pp.neighbors(self.adata, n_neighbors=15, n_pcs=30)
        sc.tl.umap(self.adata)
        
        # Create train/test split (spatial holdout)
        x_coords = self.adata.obs['x_coord'].values
        
        # Hold out middle region of each batch
        test_mask = np.zeros(self.adata.n_obs, dtype=bool)
        for batch in self.adata.obs['batch'].unique():
            batch_mask = (self.adata.obs['batch'] == batch).values
            batch_x = x_coords[batch_mask]
            x_min, x_max = batch_x.min(), batch_x.max()
            test_x_min = x_min + (x_max - x_min) * 0.4
            test_x_max = x_min + (x_max - x_min) * 0.6
            
            batch_test = batch_mask & (x_coords >= test_x_min) & (x_coords <= test_x_max)
            test_mask |= batch_test
        
        self.adata.obs['split'] = 'train'
        self.adata.obs.loc[test_mask, 'split'] = 'test'
        
        # Summary statistics
        print(f"\n  Final dataset: {self.adata.n_obs} cells, {self.adata.n_vars} genes")
        print(f"  Train: {(self.adata.obs['split']=='train').sum()}, "
              f"Test: {(self.adata.obs['split']=='test').sum()}")
        
        return self.adata
    
    def compute_batch_mixing_metrics(self) -> Dict:
        """
        Compute iLISI and cLISI to quantify batch mixing and biology preservation.
        """
        print("\nComputing batch mixing metrics...")
        
        # Use PCA embedding
        X_pca = self.adata.obsm['X_pca'][:, :30]
        
        # Compute k-nearest neighbors
        k = 30
        nn = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')
        nn.fit(X_pca)
        distances, indices = nn.kneighbors(X_pca)
        
        # Remove self from neighbors
        indices = indices[:, 1:]
        
        # Compute iLISI (batch mixing)
        batch_labels = self.adata.obs['batch'].values
        ilisi_scores = []
        
        for i in range(len(indices)):
            neighbor_batches = batch_labels[indices[i]]
            # Simpson's index
            _, counts = np.unique(neighbor_batches, return_counts=True)
            p = counts / counts.sum()
            simpson = (p ** 2).sum()
            ilisi = 1 / simpson  # Inverse Simpson's
            ilisi_scores.append(ilisi)
        
        # Compute cLISI (cell type purity)
        celltype_labels = self.adata.obs['cell_type'].values
        clisi_scores = []
        
        for i in range(len(indices)):
            neighbor_types = celltype_labels[indices[i]]
            _, counts = np.unique(neighbor_types, return_counts=True)
            p = counts / counts.sum()
            simpson = (p ** 2).sum()
            clisi = 1 / simpson
            clisi_scores.append(clisi)
        
        metrics = {
            'iLISI_mean': np.mean(ilisi_scores),
            'iLISI_median': np.median(ilisi_scores),
            'cLISI_mean': np.mean(clisi_scores),
            'cLISI_median': np.median(clisi_scores),
            'iLISI_per_cell': ilisi_scores,
            'cLISI_per_cell': clisi_scores
        }
        
        n_batches = len(np.unique(batch_labels))
        print(f"  iLISI: {metrics['iLISI_mean']:.3f} (ideal: {n_batches})")
        print(f"  cLISI: {metrics['cLISI_mean']:.3f} (ideal: 1.0)")
        
        return metrics
    
    def plot_quality_control(self, save_path: Optional[Path] = None):
        """Generate comprehensive QC plots."""
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Row 1: UMAP visualizations
        sc.pl.umap(self.adata, color='cell_type', ax=axes[0, 0], show=False,
                   title='Cell Types')
        sc.pl.umap(self.adata, color='batch', ax=axes[0, 1], show=False,
                   title='Batches (should separate)')
        sc.pl.umap(self.adata, color='is_interacting', ax=axes[0, 2], show=False,
                   title='Interacting Cells')
        
        # Library sizes by batch
        if 'counts_final' in self.adata.layers:
            lib_sizes = np.array(self.adata.layers['counts_final'].sum(axis=1)).flatten()
            self.adata.obs['library_size'] = lib_sizes
        sc.pl.umap(self.adata, color='library_size', ax=axes[0, 3], show=False,
                   title='Library Size')
        
        # Row 2: Spatial plots per batch
        batches = sorted(self.adata.obs['batch'].unique())
        for i, batch in enumerate(batches[:3]):
            batch_mask = self.adata.obs['batch'] == batch
            batch_data = self.adata[batch_mask]
            
            # Color by cell type
            cell_type_codes = pd.Categorical(batch_data.obs['cell_type']).codes
            scatter = axes[1, i].scatter(
                batch_data.obs['x_coord'],
                batch_data.obs['y_coord'],
                c=cell_type_codes,
                s=5, alpha=0.6, cmap='tab10'
            )
            axes[1, i].set_title(f'{batch} Spatial Layout')
            axes[1, i].set_xlabel('X')
            axes[1, i].set_ylabel('Y')
        
        # Overall spatial
        cell_type_codes = pd.Categorical(self.adata.obs['cell_type']).codes
        axes[1, 3].scatter(
            self.adata.obs['x_coord'],
            self.adata.obs['y_coord'],
            c=cell_type_codes,
            s=2, alpha=0.4, cmap='tab10'
        )
        axes[1, 3].set_title('All Batches Spatial')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved QC plot to {save_path}")
        
        plt.close()
    
    def save(self, output_path: Path):
        """Save the dataset."""
        import json # Import json for cleaner string conversion
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # FIX: Convert complex structures to JSON strings.
        # AnnData/HDF5 crashes on lists of dictionaries or dicts of dicts.
        # Serializing them ensures they save correctly as a single string block.
        
        # We use str() or json.dumps() to make them storable
        self.adata.uns['batch_effects'] = str(self.batch_effects_applied)
        self.adata.uns['interaction_rules'] = str(self.interaction_rules)
        self.adata.uns['batch_proportions'] = str(self.batch_proportions)
        
        self.adata.write_h5ad(output_path)
        print(f"Saved to {output_path}")


def generate_ba_amici_benchmark(
    source_path: Path,
    output_dir: Path,
    n_replicates: int = 10,
    cells_per_batch: int = 2000,
    base_seed: int = 42
):
    """
    Generate multiple benchmark replicates for BA-AMICI evaluation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading source data...")
    source_adata = sc.read_h5ad(source_path)
    
    print("\n" + "="*70)
    print("BA-AMICI BENCHMARK GENERATOR v2")
    print("Creating data with confounded batch effects and consistent interactions")
    print("="*70)
    
    for rep_id in range(n_replicates):
        print(f"\n{'='*70}")
        print(f"REPLICATE {rep_id:02d}")
        print(f"{'='*70}")
        
        gen = ProperBatchBenchmarkGenerator(
            source_adata.copy(),
            n_batches=3,
            n_cell_types=3,
            seed=base_seed + rep_id
        )
        
        # Run pipeline
        (gen
         .prepare_source_data(resolution=0.5)
         .generate_batches_with_confounding(
             cells_per_batch=cells_per_batch,
             proportion_variation=0.25
         )
         .apply_batch_effects_to_counts(
             library_size_variation=0.4,
             gene_detection_variation=0.3,
             dropout_variation=0.15
         )
         .assign_spatial_coordinates_per_batch(domain_spread=180)
         .imprint_consistent_interactions()
         .normalize_and_finalize())
        
        # Compute metrics
        metrics = gen.compute_batch_mixing_metrics()
        
        # Save
        output_path = output_dir / f"replicate_{rep_id:02d}.h5ad"
        gen.save(output_path)
        
        # Plot
        plot_path = output_dir / f"replicate_{rep_id:02d}_qc.png"
        gen.plot_quality_control(plot_path)
    
    print(f"\n{'='*70}")
    print(f"✓ Generated {n_replicates} replicates in {output_dir}")
    print("="*70)


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    
    # provide source data with raw counts
    source_file = current_dir / "pbmc_data" / "pbmc_multi_batch.h5ad"
    output_dir = current_dir / "ba_amici_benchmark_v2"
    
    if not source_file.exists():
        print(f"Source file not found: {source_file}")
        print("Please run Synthetic_Data_Prep.py first or provide source data")
    else:
        generate_ba_amici_benchmark(
            source_path=source_file,
            output_dir=output_dir,
            n_replicates=10,
            cells_per_batch=2000
        )