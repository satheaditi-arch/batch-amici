"""
Semi-synthetic Spatial Transcriptomics Benchmark Generator for BA-AMICI

This module generates semi-synthetic spatial transcriptomics data following the 
methodology described in the AMICI paper (Hong et al., 2025), extended to include
batch effects for evaluating batch-aware methods.

Key Features:
1. Subclustering cell types to create interacting vs neutral phenotypes
2. Spatial placement based on interaction rules (border-based phenotype assignment)
3. Realistic batch effects that create technical variation while preserving biology
4. Ground truth storage for downstream evaluation

Reference:
    Hong et al. (2025). AMICI: Attention Mechanism Interpretation of Cell-cell 
    Interactions. bioRxiv 2025.09.22.677860

Author: BA-AMICI Project
"""

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import pairwise_distances

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class InteractionRule:
    """Defines a ground-truth cell-cell interaction."""
    sender_type: str
    receiver_type: str
    length_scale: float  # Distance threshold in micrometers
    interaction_subtype: str  # Label for cells that received the interaction
    neutral_subtype: str  # Label for cells that did not receive the interaction


@dataclass
class BatchEffectConfig:
    """Configuration for batch-specific technical variation."""
    library_size_factor: float = 1.0  # Multiplicative factor for total counts
    dropout_rate: float = 0.0  # Additional dropout probability
    gene_detection_bias: Optional[np.ndarray] = None  # Gene-specific detection rates
    noise_scale: float = 0.0  # Additive noise standard deviation


@dataclass
class GroundTruth:
    """Storage for ground truth information used in evaluation."""
    interactions: Dict[str, InteractionRule] = field(default_factory=dict)
    interacting_cells: Dict[str, List[str]] = field(default_factory=dict)  # interaction_name -> cell_ids
    de_genes: Dict[str, pd.DataFrame] = field(default_factory=dict)  # interaction_name -> DE gene df
    batch_assignments: pd.Series = field(default_factory=pd.Series)


class SemisyntheticBatchGenerator:
    """
    Generator for semi-synthetic spatial transcriptomics data with batch effects.
    
    This class follows the AMICI paper methodology:
    1. Load source single-cell data (e.g., 68k PBMC)
    2. Cluster to identify cell types
    3. Subcluster each cell type to identify phenotypic states
    4. Assign spatial coordinates with cell type domains
    5. Sample from subclusters based on spatial proximity (interaction rules)
    6. Apply batch-specific technical effects
    
    The key insight is that interactions create PHENOTYPIC SHIFTS - cells near 
    senders are sampled from an "interacting" subcluster with genuinely different
    gene expression, not artificially scaled genes.
    
    Parameters
    ----------
    source_adata : AnnData
        Source single-cell data. Should have raw counts or be convertible.
    n_cell_types : int
        Number of major cell types to use in simulation.
    n_subclusters_per_type : int
        Number of subclusters per cell type (for phenotypic states).
    seed : int
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        source_adata: ad.AnnData,
        n_cell_types: int = 3,
        n_subclusters_per_type: int = 3,
        seed: int = 42,
    ):
        self.source = source_adata.copy()
        self.n_cell_types = n_cell_types
        self.n_subclusters_per_type = n_subclusters_per_type
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Will be populated during generation
        self.processed_source: Optional[ad.AnnData] = None
        self.cell_type_map: Dict[str, str] = {}  # cluster_id -> cell_type_label
        self.subtype_map: Dict[str, List[str]] = {}  # cell_type -> list of subtypes
        self.ground_truth = GroundTruth()
        
        self._validate_source_data()
        
    def _validate_source_data(self):
        """Validate source data has appropriate structure."""
        # Check for counts
        if sp.issparse(self.source.X):
            max_val = self.source.X.max()
        else:
            max_val = np.max(self.source.X)
            
        if max_val < 10:
            warnings.warn(
                f"Source data max value is {max_val}. Expected raw counts (typically 100s-1000s). "
                "Data may already be normalized, which could affect batch effect simulation."
            )
            
        print(f"Source data: {self.source.n_obs} cells, {self.source.n_vars} genes")
        print(f"Expression range: [{self.source.X.min():.1f}, {self.source.X.max():.1f}]")
    
    def prepare_source(
        self,
        n_hvgs: int = 500,
        use_scvi: bool = True,
        clustering_resolution: float = 0.3,
    ) -> "SemisyntheticBatchGenerator":
        """
        Prepare source data by clustering and subclustering.
        
        This follows the AMICI paper preprocessing:
        1. Normalize and log-transform
        2. Select highly variable genes
        3. Train scVI (optional) or use PCA for embeddings
        4. Leiden clustering to identify cell types
        5. Subcluster each cell type to identify phenotypic states
        
        Parameters
        ----------
        n_hvgs : int
            Number of highly variable genes to select.
        use_scvi : bool
            Whether to use scVI for embeddings. Falls back to PCA if False or unavailable.
        clustering_resolution : float
            Resolution for Leiden clustering.
            
        Returns
        -------
        self : SemisyntheticBatchGenerator
            For method chaining.
        """
        print("\n" + "="*60)
        print("Step 1: Preparing source data")
        print("="*60)
        
        adata = self.source.copy()
        
        # Store raw counts
        if sp.issparse(adata.X):
            adata.layers["counts"] = adata.X.toarray()
        else:
            adata.layers["counts"] = adata.X.copy()
        
        # Normalize
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # HVG selection
        sc.pp.highly_variable_genes(
            adata, 
            n_top_genes=n_hvgs, 
            flavor="seurat_v3",
            layer="counts"
        )
        adata = adata[:, adata.var["highly_variable"]].copy()
        print(f"  Selected {adata.n_vars} highly variable genes")
        
        # Embedding
        if use_scvi:
            try:
                import scvi as scvi_module
                print("  Training scVI model for embeddings...")
                scvi_module.model.SCVI.setup_anndata(adata, layer="counts")
                model = scvi_module.model.SCVI(adata)
                model.train(max_epochs=100, early_stopping=True)
                adata.obsm["X_embedding"] = model.get_latent_representation()
                print("  scVI training complete")
            except (ImportError, Exception) as e:
                print(f"  scVI unavailable ({e}), using PCA")
                sc.pp.pca(adata, n_comps=50)
                adata.obsm["X_embedding"] = adata.obsm["X_pca"]
        else:
            sc.pp.pca(adata, n_comps=50)
            adata.obsm["X_embedding"] = adata.obsm["X_pca"]
            
        # Clustering
        print("  Performing Leiden clustering...")
        sc.pp.neighbors(adata, use_rep="X_embedding")
        sc.tl.leiden(adata, resolution=clustering_resolution, key_added="leiden")
        
        # Select top N clusters by size
        cluster_sizes = adata.obs["leiden"].value_counts()
        top_clusters = cluster_sizes.nlargest(self.n_cell_types).index.tolist()
        
        # Filter to keep only these clusters
        mask = adata.obs["leiden"].isin(top_clusters)
        adata = adata[mask].copy()
        
        # Map cluster IDs to cell type labels (A, B, C, ...)
        self.cell_type_map = {
            cluster: chr(65 + i)  # 65 = ASCII 'A'
            for i, cluster in enumerate(top_clusters)
        }
        adata.obs["cell_type"] = adata.obs["leiden"].map(self.cell_type_map)
        
        print(f"  Retained {adata.n_obs} cells across {self.n_cell_types} cell types:")
        for ct in sorted(adata.obs["cell_type"].unique()):
            n = (adata.obs["cell_type"] == ct).sum()
            print(f"    Type {ct}: {n} cells")
        
        self.processed_source = adata
        return self
    
    def subcluster_cell_types(
        self,
        n_genes_for_subclustering: int = 50,
    ) -> "SemisyntheticBatchGenerator":
        """
        Subcluster each cell type to identify phenotypic states.
        
        This is critical for the AMICI methodology: subclusters represent
        different biological states within a cell type. One subcluster will
        be designated as "interacting" (cells that received signal from sender)
        and another as "neutral" (cells that did not receive signal).
        
        Parameters
        ----------
        n_genes_for_subclustering : int
            Number of genes to use for subclustering (mix of highly expressed 
            and random genes).
            
        Returns
        -------
        self : SemisyntheticBatchGenerator
            For method chaining.
        """
        if self.processed_source is None:
            raise ValueError("Must call prepare_source() first")
            
        print("\n" + "="*60)
        print("Step 2: Subclustering cell types")
        print("="*60)
        
        adata = self.processed_source
        adata.obs["subtype"] = "unassigned"
        
        for cell_type in adata.obs["cell_type"].unique():
            print(f"\n  Subclustering type {cell_type}...")
            
            # Select genes for subclustering
            # Use mix of highly expressed and random genes
            gene_means = np.ravel(np.mean(adata.X, axis=0))
            sorted_indices = np.argsort(gene_means)
            
            n_top = n_genes_for_subclustering // 2
            n_random = n_genes_for_subclustering - n_top
            
            top_genes = sorted_indices[-n_top:]
            remaining = sorted_indices[:-n_top]
            random_genes = self.rng.choice(remaining, n_random, replace=False)
            selected_genes = np.concatenate([top_genes, random_genes])
            
            # Create subset for subclustering
            type_mask = adata.obs["cell_type"] == cell_type
            type_indices = adata.obs_names[type_mask]
            adata_sub = adata[type_indices, selected_genes].copy()
            
            # PCA on subset
            sc.pp.pca(adata_sub, n_comps=min(20, len(selected_genes) - 1))
            adata.obsm[f"X_subcluster_{cell_type}"] = np.zeros((adata.n_obs, adata_sub.obsm["X_pca"].shape[1]))
            adata.obsm[f"X_subcluster_{cell_type}"][type_mask] = adata_sub.obsm["X_pca"]
            
            # Binary search for resolution to get target number of subclusters
            target_n = self.n_subclusters_per_type
            subclusters = self._leiden_binary_search(
                adata_sub, 
                target_n, 
                key_added=f"subcluster_{cell_type}"
            )
            
            # Map subclusters to labels
            subtype_labels = [f"{cell_type}_sub{i}" for i in range(len(subclusters))]
            subcluster_map = dict(zip(subclusters, subtype_labels))
            
            adata.obs.loc[type_mask, "subtype"] = (
                adata_sub.obs[f"subcluster_{cell_type}"].map(subcluster_map).values
            )
            
            self.subtype_map[cell_type] = subtype_labels
            print(f"    Created {len(subtype_labels)} subclusters: {subtype_labels}")
        
        adata.obs["subtype"] = adata.obs["subtype"].astype("category")
        self.processed_source = adata
        return self
    
    def _leiden_binary_search(
        self,
        adata: ad.AnnData,
        target_n: int,
        key_added: str,
        min_resolution: float = 0.01,
        max_resolution: float = 5.0,
        max_iterations: int = 50,
    ) -> List[str]:
        """
        Binary search to find Leiden resolution that gives target number of clusters.
        
        Parameters
        ----------
        adata : AnnData
            Data to cluster.
        target_n : int
            Target number of clusters.
        key_added : str
            Key to store cluster assignments in obs.
        min_resolution, max_resolution : float
            Range for binary search.
        max_iterations : int
            Maximum iterations for search.
            
        Returns
        -------
        clusters : List[str]
            List of cluster labels found.
        """
        sc.pp.neighbors(adata, use_rep="X_pca")
        
        for _ in range(max_iterations):
            resolution = (min_resolution + max_resolution) / 2
            sc.tl.leiden(adata, resolution=resolution, key_added=key_added)
            n_clusters = len(adata.obs[key_added].unique())
            
            if n_clusters == target_n:
                break
            elif n_clusters < target_n:
                min_resolution = resolution + 0.01
            else:
                max_resolution = resolution - 0.01
                
            if max_resolution <= min_resolution:
                break
        
        clusters = sorted(adata.obs[key_added].unique().tolist())
        return clusters
    
    def define_interactions(
        self,
        interactions: Dict[str, Dict],
    ) -> "SemisyntheticBatchGenerator":
        """
        Define ground-truth cell-cell interactions.
        
        Parameters
        ----------
        interactions : Dict[str, Dict]
            Dictionary mapping interaction names to configurations.
            Each configuration should have:
            - sender: sender cell type (e.g., "A")
            - receiver: receiver cell type (e.g., "B")
            - length_scale: interaction distance in micrometers
            
        -------
        >>> generator.define_interactions({
        ...     "A_to_B": {"sender": "A", "receiver": "B", "length_scale": 10},
        ...     "C_to_A": {"sender": "C", "receiver": "A", "length_scale": 20},
        ... })
        
        Returns
        -------
        self : SemisyntheticBatchGenerator
            For method chaining.
        """
        print("\n" + "="*60)
        print("Step 3: Defining interaction rules")
        print("="*60)
        
        for name, config in interactions.items():
            sender = config["sender"]
            receiver = config["receiver"]
            length_scale = config["length_scale"]
            
            # Get subclusters for receiver type
            receiver_subtypes = self.subtype_map.get(receiver, [])
            if len(receiver_subtypes) < 2:
                raise ValueError(
                    f"Receiver type {receiver} needs at least 2 subclusters. "
                    f"Found: {receiver_subtypes}"
                )
            
            # Designate first subcluster as neutral, second as interacting
            neutral_subtype = receiver_subtypes[0]
            interaction_subtype = receiver_subtypes[1]
            
            rule = InteractionRule(
                sender_type=sender,
                receiver_type=receiver,
                length_scale=length_scale,
                interaction_subtype=interaction_subtype,
                neutral_subtype=neutral_subtype,
            )
            
            self.ground_truth.interactions[name] = rule
            print(f"  {name}: {sender} → {receiver} (radius={length_scale}μm)")
            print(f"    Neutral: {neutral_subtype}, Interacting: {interaction_subtype}")
        
        return self
    
    def verify_differential_expression(
        self,
        lfc_threshold: float = 0.2,
        pval_threshold: float = 0.05,
    ) -> "SemisyntheticBatchGenerator":
        """
        Verify that interaction subtypes have differential expression vs neutral.
        
        This is a sanity check to ensure the subclusters we designated as
        "interacting" actually have different gene expression from "neutral".
        
        Parameters
        ----------
        lfc_threshold : float
            Log fold-change threshold for DE genes.
        pval_threshold : float
            P-value threshold for DE genes.
            
        Returns
        -------
        self : SemisyntheticBatchGenerator
            For method chaining.
        """
        print("\n" + "="*60)
        print("Step 4: Verifying differential expression")
        print("="*60)
        
        adata = self.processed_source
        
        for name, rule in self.ground_truth.interactions.items():
            print(f"\n  Interaction: {name}")
            
            # Get cells from the two subtypes
            mask = adata.obs["subtype"].isin([rule.neutral_subtype, rule.interaction_subtype])
            adata_de = adata[mask].copy()
            
            # Run DE analysis
            sc.tl.rank_genes_groups(
                adata_de,
                groupby="subtype",
                method="t-test",
                reference=rule.neutral_subtype,
            )
            
            # Extract results for interaction subtype
            de_results = sc.get.rank_genes_groups_df(
                adata_de, 
                group=rule.interaction_subtype
            )
            
            # Count significant DE genes
            sig_up = de_results[
                (de_results["logfoldchanges"] > lfc_threshold) &
                (de_results["pvals_adj"] < pval_threshold)
            ]
            sig_down = de_results[
                (de_results["logfoldchanges"] < -lfc_threshold) &
                (de_results["pvals_adj"] < pval_threshold)
            ]
            
            print(f"    Upregulated genes: {len(sig_up)}")
            print(f"    Downregulated genes: {len(sig_down)}")
            
            if len(sig_up) < 5:
                warnings.warn(
                    f"Interaction {name} has few upregulated DE genes ({len(sig_up)}). "
                    "Consider using different subclusters or adjusting thresholds."
                )
            
            # Store for ground truth
            de_results["class"] = 0
            de_results.loc[
                (de_results["logfoldchanges"] > lfc_threshold) &
                (de_results["pvals_adj"] < pval_threshold),
                "class"
            ] = 1
            
            self.ground_truth.de_genes[name] = de_results
        
        return self
    
    def generate_spatial_layout(
        self,
        n_cells: int = 20000,
        spatial_extent: Tuple[float, float] = (2000, 1000),
        gradient_width: float = 0.2,
    ) -> pd.DataFrame:
        """
        Generate spatial coordinates with cell type domains.
        
        Creates a spatial layout where cell types occupy different quadrants
        with gradient transitions at boundaries. This allows interaction
        zones where different cell types are adjacent.
        
        Parameters
        ----------
        n_cells : int
            Total number of cells to generate.
        spatial_extent : Tuple[float, float]
            (width, height) of spatial domain in micrometers.
        gradient_width : float
            Width of gradient transition zone as fraction of domain.
            
        Returns
        -------
        spatial_df : pd.DataFrame
            DataFrame with columns: Cell_ID, X, Y, Cell_Type, Subtype
        """
        print("\n" + "="*60)
        print("Step 5: Generating spatial layout")
        print("="*60)
        
        width, height = spatial_extent
        
        # Generate uniform random positions
        positions = self.rng.random((n_cells, 2))
        positions[:, 0] *= width
        positions[:, 1] *= height
        
        # Get cell types
        cell_types = list(self.subtype_map.keys())
        if len(cell_types) != 3:
            raise ValueError(f"Expected 3 cell types, got {len(cell_types)}")
        
        ct1, ct2, ct3 = cell_types
        
        # Assign cell types based on spatial position
        # Using triangular gradient pattern from AMICI paper
        mid_x = width / 2
        mid_y = height / 2
        
        assigned_types = []
        for pos in positions:
            x, y = pos
            
            # Gradient probabilities
            x_grad = abs(x - mid_x) / (width / 2) * (1 - gradient_width)
            y_grad = abs(y - mid_y) / (height / 2) * (1 - gradient_width)
            
            if x < mid_x:  # Left half
                if y < mid_y:  # Bottom left
                    ct = ct1 if self.rng.random() < (1 - x_grad) else ct3
                else:  # Top left
                    ct = ct1 if self.rng.random() < (1 - y_grad) else ct2
            else:  # Right half
                if y < mid_y:  # Bottom right
                    ct = ct3 if self.rng.random() < (1 - x_grad) else ct1
                else:  # Top right
                    ct = ct2 if self.rng.random() < (1 - y_grad) else ct3
            
            assigned_types.append(ct)
        
        # Create DataFrame
        spatial_df = pd.DataFrame({
            "Cell_ID": range(n_cells),
            "X": positions[:, 0],
            "Y": positions[:, 1],
            "Cell_Type": assigned_types,
        })
        
        # Initialize subtype as neutral
        for ct in cell_types:
            neutral_subtype = self.subtype_map[ct][0]  # First subcluster is neutral
            spatial_df.loc[spatial_df["Cell_Type"] == ct, "Subtype"] = neutral_subtype
        
        print(f"  Generated {n_cells} cells in {width}x{height}μm domain")
        for ct in cell_types:
            n = (spatial_df["Cell_Type"] == ct).sum()
            print(f"    Type {ct}: {n} cells")
        
        return spatial_df
    
    def apply_interaction_rules(
        self,
        spatial_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Assign interacting subtypes based on spatial proximity to senders.
        
        For each interaction rule, cells of the receiver type that are within
        the length_scale of any sender cell get assigned the "interacting"
        subtype instead of "neutral".
        
        Parameters
        ----------
        spatial_df : pd.DataFrame
            Spatial layout from generate_spatial_layout().
            
        Returns
        -------
        spatial_df : pd.DataFrame
            Updated DataFrame with Subtype reflecting interactions.
        """
        print("\n" + "="*60)
        print("Step 6: Applying interaction rules")
        print("="*60)
        
        coords = spatial_df[["X", "Y"]].values
        
        for name, rule in self.ground_truth.interactions.items():
            print(f"\n  Applying {name}: {rule.sender_type} → {rule.receiver_type}")
            
            # Get indices of potential receivers (currently neutral)
            receiver_mask = (
                (spatial_df["Cell_Type"] == rule.receiver_type) &
                (spatial_df["Subtype"] == rule.neutral_subtype)
            )
            receiver_indices = spatial_df[receiver_mask].index.tolist()
            
            # Get indices of senders
            sender_mask = spatial_df["Cell_Type"] == rule.sender_type
            sender_indices = spatial_df[sender_mask].index.tolist()
            
            if len(receiver_indices) == 0 or len(sender_indices) == 0:
                print(f"    Warning: No receivers or senders found")
                continue
            
            # Compute distances from receivers to senders
            receiver_coords = coords[receiver_indices]
            sender_coords = coords[sender_indices]
            distances = pairwise_distances(receiver_coords, sender_coords)
            
            # Find receivers within range of any sender
            min_distances = distances.min(axis=1)
            interacting_mask = min_distances <= rule.length_scale
            interacting_indices = np.array(receiver_indices)[interacting_mask]
            
            # Update subtype
            spatial_df.loc[interacting_indices, "Subtype"] = rule.interaction_subtype
            
            # Store for ground truth
            self.ground_truth.interacting_cells[name] = list(interacting_indices)
            
            n_interacting = len(interacting_indices)
            n_total = len(receiver_indices)
            print(f"    {n_interacting}/{n_total} receivers within {rule.length_scale}μm of sender")
        
        return spatial_df
    
    def sample_expression(
        self,
        spatial_df: pd.DataFrame,
    ) -> ad.AnnData:
        """
        Sample gene expression from source data based on assigned subtypes.
        
        Each cell in the spatial layout gets expression sampled from the
        appropriate subcluster in the source data. This is the key mechanism
        by which interactions create phenotypic shifts.
        
        Parameters
        ----------
        spatial_df : pd.DataFrame
            Spatial layout with Cell_Type and Subtype assignments.
            
        Returns
        -------
        adata : AnnData
            Spatial transcriptomics data with sampled expression.
        """
        print("\n" + "="*60)
        print("Step 7: Sampling expression from subclusters")
        print("="*60)
        
        source = self.processed_source
        sampled_adatas = []
        
        for subtype in spatial_df["Subtype"].unique():
            # Get source cells of this subtype
            source_mask = source.obs["subtype"] == subtype
            source_indices = source.obs_names[source_mask].tolist()
            
            # Get spatial cells assigned to this subtype
            spatial_mask = spatial_df["Subtype"] == subtype
            n_samples = spatial_mask.sum()
            
            if len(source_indices) == 0:
                raise ValueError(f"No source cells for subtype {subtype}")
            
            # Sample with replacement
            sampled_indices = self.rng.choice(source_indices, n_samples, replace=True)
            
            # Create AnnData for this subtype
            sampled = source[sampled_indices].copy()
            
            # Assign spatial coordinates
            spatial_subset = spatial_df[spatial_mask]
            sampled.obsm["spatial"] = spatial_subset[["X", "Y"]].values
            sampled.obs["Cell_ID"] = spatial_subset["Cell_ID"].values
            sampled.obs["cell_type"] = spatial_subset["Cell_Type"].values
            sampled.obs["subtype"] = spatial_subset["Subtype"].values
            
            sampled_adatas.append(sampled)
            print(f"  {subtype}: sampled {n_samples} cells")
        
        # Concatenate
        adata = ad.concat(sampled_adatas, axis=0)
        adata.obs_names_make_unique()
        
        # Sort by Cell_ID to maintain spatial order
        adata = adata[adata.obs.sort_values("Cell_ID").index].copy()
        
        print(f"\n  Total: {adata.n_obs} cells, {adata.n_vars} genes")
        
        return adata
    
    def apply_batch_effects(
        self,
        adata: ad.AnnData,
        n_batches: int = 3,
        batch_configs: Optional[List[BatchEffectConfig]] = None,
    ) -> ad.AnnData:
        """
        Apply batch effects using Splatter's proven approach:
        - Log-normal multiplicative factors per gene per batch
        - Applied to raw counts BEFORE normalization
        """
        print("\n" + "="*60)
        print("Step 8: Applying batch effects (Splatter method)")
        print("="*60)
        
        # Assign cells to batches
        n_cells = adata.n_obs
        batch_assignments = np.repeat(np.arange(n_batches), n_cells // n_batches + 1)[:n_cells]
        self.rng.shuffle(batch_assignments)
        adata.obs["batch"] = [f"batch_{i}" for i in batch_assignments]
        adata.obs["batch_id"] = batch_assignments
        
        # Get raw counts
        if "counts" in adata.layers:
            X = adata.layers["counts"].copy()
        else:
            X = adata.X.copy()
        
        if sp.issparse(X):
            X = X.toarray()
        X = X.astype(np.float64)
        
        n_genes = X.shape[1]
        
        # =========================================================
        # SPLATTER METHOD: Log-normal multiplicative factors
        # =========================================================
        # Key parameters (from Splatter defaults, increased for visibility)
        batch_fac_loc = 0.1    # meanlog of log-normal
        batch_fac_scale = 0.4  # sdlog of log-normal (controls batch effect strength!)
        
        # Generate batch factors for each gene in each batch
        # Factor = exp(N(loc, scale))
        # This gives factors centered around 1 with some spread
        
        for batch_idx in range(n_batches):
            batch_mask = batch_assignments == batch_idx
            n_batch_cells = batch_mask.sum()
            
            # Sample log-normal factors for this batch
            # Using different random factors per batch creates batch-specific patterns
            log_factors = self.rng.normal(
                loc=batch_fac_loc, 
                scale=batch_fac_scale, 
                size=n_genes
            )
            batch_factors = np.exp(log_factors)
            
            # Apply multiplicative factors to all cells in this batch
            # This is the key step - multiply raw counts by gene-specific batch factors
            X[batch_mask] = X[batch_mask] * batch_factors[np.newaxis, :]
            
            print(f"\n  Batch {batch_idx} ({n_batch_cells} cells):")
            print(f"    Factor range: [{batch_factors.min():.3f}, {batch_factors.max():.3f}]")
            print(f"    Factor mean: {batch_factors.mean():.3f}")
            print(f"    Factor std: {batch_factors.std():.3f}")
        
        # Store batch-affected counts
        adata.layers["counts_batch"] = X.copy()
        
        # Normalize (batch effects will persist because they changed relative proportions)
        adata.X = X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # Verify batch effects
        print("\n  Verification (post-normalization):")
        batch_means = []
        for b in range(n_batches):
            mask = batch_assignments == b
            batch_means.append(np.mean(adata.X[mask], axis=0))
        
        corrs = []
        for i in range(n_batches):
            for j in range(i+1, n_batches):
                c = np.corrcoef(batch_means[i], batch_means[j])[0, 1]
                corrs.append(c)
        
        mean_corr = np.mean(corrs)
        print(f"Batch profile correlation: {mean_corr:.4f}")
        
        if mean_corr < 0.95:
            print("Good batch separation!")
        else:
            print("Increase batch_fac_scale for stronger effects")
        
        return adata

    def create_train_test_split(
        self,
        adata: ad.AnnData,
        test_x_range: Tuple[float, float] = (900, 1100),
    ) -> ad.AnnData:
        """
        Create train/test split based on spatial location.
        
        Following AMICI paper: test set is a spatial slice to ensure
        it contains all neighborhood types.
        
        Parameters
        ----------
        adata : AnnData
            Data with spatial coordinates.
        test_x_range : Tuple[float, float]
            X-coordinate range for test set.
            
        Returns
        -------
        adata : AnnData
            Data with train_test_split column in obs.
        """
        x_coords = adata.obsm["spatial"][:, 0]
        test_mask = (x_coords >= test_x_range[0]) & (x_coords <= test_x_range[1])
        
        adata.obs["train_test_split"] = "train"
        adata.obs.loc[test_mask, "train_test_split"] = "test"
        
        n_train = (adata.obs["train_test_split"] == "train").sum()
        n_test = (adata.obs["train_test_split"] == "test").sum()
        print(f"\n  Train/test split: {n_train} train, {n_test} test")
        
        return adata
    
    def generate(
        self,
        interactions: Dict[str, Dict],
        n_cells: int = 20000,
        n_batches: int = 3,
        spatial_extent: Tuple[float, float] = (2000, 1000),
        batch_configs: Optional[List[BatchEffectConfig]] = None,
    ) -> Tuple[ad.AnnData, GroundTruth]:
        """
        Run the complete generation pipeline.
        
        Parameters
        ----------
        interactions : Dict[str, Dict]
            Interaction definitions. See define_interactions().
        n_cells : int
            Number of cells to generate.
        n_batches : int
            Number of batches.
        spatial_extent : Tuple[float, float]
            Spatial domain size in micrometers.
        batch_configs : List[BatchEffectConfig], optional
            Batch effect configurations.
            
        Returns
        -------
        adata : AnnData
            Generated semi-synthetic data.
        ground_truth : GroundTruth
            Ground truth information for evaluation.
        """
        # Define interactions
        self.define_interactions(interactions)
        
        # Verify DE between subtypes
        self.verify_differential_expression()
        
        # Generate spatial layout
        spatial_df = self.generate_spatial_layout(
            n_cells=n_cells,
            spatial_extent=spatial_extent,
        )
        
        # Apply interaction rules
        spatial_df = self.apply_interaction_rules(spatial_df)
        
        # Sample expression
        adata = self.sample_expression(spatial_df)
        
        # Apply batch effects
        adata = self.apply_batch_effects(
            adata,
            n_batches=n_batches,
            batch_configs=batch_configs,
        )
        
        # Train/test split
        adata = self.create_train_test_split(adata)
        
        # Final statistics
        print("\n" + "="*60)
        print("Generation complete!")
        print("="*60)
        print(f"  Cells: {adata.n_obs}")
        print(f"  Genes: {adata.n_vars}")
        print(f"  Batches: {n_batches}")
        print(f"  Interactions: {len(self.ground_truth.interactions)}")
        
        return adata, self.ground_truth


def generate_replicate(
    source_path: Union[str, Path],
    output_path: Union[str, Path],
    seed: int,
    interactions: Dict[str, Dict],
    n_cells: int = 20000,
    n_batches: int = 3,
    n_hvgs: int = 500,
    use_scvi: bool = True,
) -> None:
    """
    Generate a single semi-synthetic replicate.
    
    Convenience function for generating multiple replicates with different seeds.
    
    Parameters
    ----------
    source_path : str or Path
        Path to source single-cell data.
    output_path : str or Path
        Path to save generated data.
    seed : int
        Random seed for this replicate.
    interactions : Dict
        Interaction definitions.
    n_cells : int
        Number of cells.
    n_batches : int
        Number of batches.
    n_hvgs : int
        Number of highly variable genes.
    use_scvi : bool
        Whether to use scVI for embeddings.
    """
    # Load source data
    source_adata = ad.read_h5ad(source_path)
    
    # Create generator
    generator = SemisyntheticBatchGenerator(
        source_adata,
        n_cell_types=3,
        n_subclusters_per_type=3,
        seed=seed,
    )
    
    # Prepare and generate
    generator.prepare_source(n_hvgs=n_hvgs, use_scvi=use_scvi)
    generator.subcluster_cell_types()
    
    adata, ground_truth = generator.generate(
        interactions=interactions,
        n_cells=n_cells,
        n_batches=n_batches,
    )
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)
    
    # Save ground truth
    gt_path = output_path.with_suffix(".ground_truth.pkl")
    import pickle
    with open(gt_path, "wb") as f:
        pickle.dump(ground_truth, f)
    
    print(f"\nSaved to {output_path}")
    print(f"Ground truth saved to {gt_path}")


if __name__ == "__main__":
    print("Semi-synthetic Batch Generator for BA-AMICI")
    print("=" * 60)
    
    # Default interactions
    DEFAULT_INTERACTIONS = {
        "A_to_B": {
            "sender": "A",
            "receiver": "B", 
            "length_scale": 10,
        },
        "C_to_A": {
            "sender": "C",
            "receiver": "A",
            "length_scale": 20,
        },
    }
    
    print("\nTo generate data, use:")
    print("  generator = SemisyntheticBatchGenerator(source_adata)")
    print("  generator.prepare_source()")
    print("  generator.subcluster_cell_types()")
    print("  adata, gt = generator.generate(interactions=DEFAULT_INTERACTIONS)")

def sample_batch_configs(
    n_batches, 
    library_size_mean=1.0, 
    library_size_std=0.5,
    dropout_mean=0.10, 
    dropout_std=0.05, 
    noise_mean=0.8,
    noise_std=0.1, 
    seed=None
):
    """Sample random batch effect configurations."""
    rng = np.random.default_rng(seed)
    configs = []
    for i in range(n_batches):
        configs.append(BatchEffectConfig(
            library_size_factor=np.clip(rng.normal(library_size_mean, library_size_std), 0.5, 2.0),
            dropout_rate=np.clip(rng.normal(dropout_mean + i*0.03, dropout_std), 0.05, 0.3),
            noise_scale=np.clip(rng.normal(noise_mean, noise_std), 0.2, 0.6),
        ))
    return configs