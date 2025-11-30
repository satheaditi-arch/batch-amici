"""
BA-AMICI Semi-Synthetic Data Generation
Based on AMICI paper methodology but with multiple batches
"""

import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt

class BASemiSyntheticGenerator:
    """
    Generate semi-synthetic spatial transcriptomics data with:
    1. Ground truth cell-cell interactions
    2. Multiple batches with batch effects
    3. Spatial organization
    """
    
    def __init__(self, adata, n_cell_types=3, random_state=42):
        """
        Initialize generator
        
        Parameters:
        -----------
        adata : AnnData
            Pre-processed PBMC data with multiple batches
        n_cell_types : int
            Number of cell types to simulate
        random_state : int
            Random seed for reproducibility
        """
        self.adata = adata
        self.n_cell_types = n_cell_types
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Store batch information
        self.batches = adata.obs['batch'].unique()
        self.n_batches = len(self.batches)
        
        print(f"Initialized generator with {self.n_batches} batches: {list(self.batches)}")
    
    def create_cell_type_clusters(self):
        """
        Create cell type clusters using scVI + Leiden clustering
        Similar to AMICI paper methodology
        """
        print("\nCreating cell type clusters...")
        
        # Use existing leiden clustering if available, otherwise create
        if 'leiden' not in self.adata.obs.columns:
            sc.tl.leiden(self.adata, resolution=0.5)
        
        # Select top N clusters as cell types
        cluster_counts = self.adata.obs['leiden'].value_counts()
        top_clusters = cluster_counts.nlargest(self.n_cell_types * 2).index.tolist()
        
        # Assign cell types A, B, C
        cell_type_mapping = {}
        cell_types = ['A', 'B', 'C']
        
        for i, cell_type in enumerate(cell_types):
            # Use pairs of clusters for each cell type
            # One for interacting, one for non-interacting
            cell_type_mapping[top_clusters[i*2]] = f'{cell_type}_interacting'
            cell_type_mapping[top_clusters[i*2 + 1]] = f'{cell_type}_noninteracting'
        
        # Map to simplified cell types
        self.adata.obs['detailed_type'] = self.adata.obs['leiden'].map(
            lambda x: cell_type_mapping.get(x, 'other')
        )
        self.adata.obs['cell_type'] = self.adata.obs['detailed_type'].map(
            lambda x: x.split('_')[0] if '_' in x else x
        )
        
        # Filter to only main cell types
        self.adata = self.adata[self.adata.obs['cell_type'].isin(cell_types)]
        
        print(f"Created {self.n_cell_types} cell types:")
        print(self.adata.obs['cell_type'].value_counts())
        print("\nDetailed types (interacting vs non-interacting):")
        print(self.adata.obs['detailed_type'].value_counts())
        
        return self
    
    def define_ground_truth_interactions(self):
        """
        Define ground truth interactions:
        - A → B at 10μm
        - C → A at 20μm
        """
        self.interactions = [
            {
                'sender': 'A',
                'receiver': 'B',
                'length_scale': 10.0,  # micrometers
                'name': 'A_to_B'
            },
            {
                'sender': 'C',
                'receiver': 'A',
                'length_scale': 20.0,  # micrometers
                'name': 'C_to_A'
            }
        ]
        
        print("\nDefined ground truth interactions:")
        for interaction in self.interactions:
            print(f"  {interaction['sender']} → {interaction['receiver']} "
                  f"(length scale: {interaction['length_scale']}μm)")
        
        return self
    
    def generate_spatial_coordinates(self, width=2000, height=1000):
        """
        Generate spatial coordinates with gradient pattern
        Similar to AMICI paper - cells arranged in quadrants with gradients
        """
        print(f"\nGenerating spatial coordinates ({width}x{height}μm)...")
        
        n_cells = self.adata.n_obs
        
        # Generate enough spatial points
        n_points = int(n_cells * 1.5)  # Generate 50% more points than needed
        x = np.random.uniform(0, width, n_points)
        y = np.random.uniform(0, height, n_points)
        
        # Randomly sample n_cells points
        indices = np.random.choice(len(x), size=n_cells, replace=False)
        x = x[indices]
        y = y[indices]
        
        # Assign cell types based on spatial location with gradients
        cell_types = []
        
        for i in range(n_cells):
            xi, yi = x[i], y[i]
            
            # Define quadrants with gradient transitions
            if xi < width/3:
                # Left third - mostly A
                if np.random.random() < 0.8:
                    cell_types.append('A')
                else:
                    cell_types.append('B' if yi > height/2 else 'C')
            elif xi < 2*width/3:
                # Middle third - mostly B (top) and C (bottom)
                if yi > height/2:
                    cell_types.append('B')
                else:
                    cell_types.append('C')
            else:
                # Right third - mixed
                cell_types.append(np.random.choice(['A', 'B', 'C']))
        
        # Store coordinates
        self.adata.obs['x'] = x
        self.adata.obs['y'] = y
        self.adata.obs['assigned_type'] = cell_types
        
        print(f"Assigned spatial cell types:")
        print(pd.Series(cell_types).value_counts())
        
        return self
    
    def assign_interacting_phenotypes(self):
        """
        Assign interacting vs non-interacting phenotypes based on:
        1. Assigned cell type from spatial location
        2. Distance to relevant sender cells
        3. Ground truth interaction length scales
        """
        print("\nAssigning interacting phenotypes...")
        
        # Calculate pairwise distances
        coords = self.adata.obs[['x', 'y']].values
        from scipy.spatial.distance import cdist
        distances = cdist(coords, coords, metric='euclidean')
        
        # For each cell, determine if it should be interacting phenotype
        interacting_status = []
        
        for i in range(self.adata.n_obs):
            cell_type = self.adata.obs['assigned_type'].iloc[i]
            is_interacting = False
            
            # Check each interaction
            for interaction in self.interactions:
                if cell_type == interaction['receiver']:
                    # Find sender cells within length scale
                    sender_mask = self.adata.obs['assigned_type'] == interaction['sender']
                    sender_indices = np.where(sender_mask)[0]
                    
                    if len(sender_indices) > 0:
                        min_distance = distances[i, sender_indices].min()
                        if min_distance <= interaction['length_scale']:
                            is_interacting = True
                            break
            
            interacting_status.append(is_interacting)
        
        self.adata.obs['is_interacting'] = interacting_status
        
        # Now sample actual cells from the appropriate phenotype
        sampled_cells = []
        
        for i in range(self.adata.n_obs):
            cell_type = self.adata.obs['assigned_type'].iloc[i]
            is_interacting = self.adata.obs['is_interacting'].iloc[i]
            
            # Get appropriate cluster
            if is_interacting:
                target_cluster = f"{cell_type}_interacting"
            else:
                target_cluster = f"{cell_type}_noninteracting"
            
            # Sample a cell from this cluster
            cluster_mask = self.adata.obs['detailed_type'] == target_cluster
            
            if cluster_mask.sum() > 0:
                sampled_idx = np.random.choice(np.where(cluster_mask)[0])
                sampled_cells.append(sampled_idx)
            else:
                # Fallback: use any cell of this type
                type_mask = self.adata.obs['cell_type'] == cell_type
                sampled_idx = np.random.choice(np.where(type_mask)[0])
                sampled_cells.append(sampled_idx)
        
        # Create new dataset with sampled cells but keeping spatial coords
        spatial_info = self.adata.obs[['x', 'y', 'assigned_type', 'is_interacting', 'batch']].copy()
        self.adata = self.adata[sampled_cells].copy()
        
        # Make obs_names unique to avoid warnings
        self.adata.obs_names_make_unique()
        
        # Restore spatial information
        self.adata.obs['x'] = spatial_info['x'].values
        self.adata.obs['y'] = spatial_info['y'].values
        self.adata.obs['cell_type'] = spatial_info['assigned_type'].values
        self.adata.obs['is_interacting'] = spatial_info['is_interacting'].values
        self.adata.obs['batch'] = spatial_info['batch'].values
        
        print(f"Interacting cells: {sum(interacting_status)} / {len(interacting_status)}")
        
        return self
    
    def introduce_batch_effects(self, batch_effect_strength=0.5):
        """
        Introduce systematic batch effects to gene expression
        """
        print(f"\nIntroducing batch effects (strength={batch_effect_strength})...")
        
        # Convert to dense array if sparse
        import scipy.sparse as sp
        if sp.issparse(self.adata.X):
            self.adata.X = self.adata.X.toarray()
        
        n_genes = self.adata.n_vars

        # For each batch, add systematic bias
        for i, batch in enumerate(self.batches):
            batch_mask = (self.adata.obs['batch'] == batch).values  # Convert to numpy array
            
            # --- FIX: Define Shift Direction ---
            if i == 0:
                shift = 1.0 * batch_effect_strength
            elif i == 1:
                shift = -1.0 * batch_effect_strength
            else:
                shift = 0.0 
            
            # --- FIX: Create Noise Vector ---
            # Mean shift + random variance
            noise_vector = np.full(n_genes, shift) + np.random.normal(0, 0.2, n_genes)
            
            # --- FIX: Apply Additive Shift (NO scale_factors) ---
            self.adata.X[batch_mask, :] = self.adata.X[batch_mask, :] + noise_vector
            
            # Ensure no negative values
            self.adata.X[self.adata.X < 0] = 0
            
            print(f"  Applied STRONG additive batch effects to {batch}")
        
        return self
    
    def split_train_test(self, test_size=0.2):
        """Split data into train and test sets"""
        print(f"\nSplitting data (test_size={test_size})...")
        
        # Split based on spatial region (as in AMICI paper)
        # Test set: cells with 900 < x < 1100
        test_mask = (self.adata.obs['x'] > 900) & (self.adata.obs['x'] < 1100)
        train_mask = ~test_mask
        
        self.adata.obs['split'] = 'train'
        self.adata.obs.loc[test_mask, 'split'] = 'test'
        
        print(f"Train cells: {train_mask.sum()}")
        print(f"Test cells: {test_mask.sum()}")
        
        return self
    
    def save(self, output_path="semi_synthetic_data.h5ad"):
        """Save generated data"""
        print(f"\nSaving to {output_path}...")
        self.adata.write(output_path)
        
        # Also save ground truth information
        gt_info = {
            'interactions': self.interactions,
            'n_batches': self.n_batches,
            'batches': list(self.batches)
        }
        
        import json
        gt_path = Path(output_path).with_suffix('.json')
        with open(gt_path, 'w') as f:
            json.dump(gt_info, f, indent=2)
        
        print(f"Ground truth info saved to {gt_path}")
        
        return self
    
    def visualize(self, save_path="semi_synthetic_viz.png"):
        """Visualize the generated data"""
        print("\nGenerating visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Spatial distribution by cell type
        ax = axes[0, 0]
        for cell_type in ['A', 'B', 'C']:
            mask = self.adata.obs['cell_type'] == cell_type
            ax.scatter(
                self.adata.obs.loc[mask, 'x'],
                self.adata.obs.loc[mask, 'y'],
                label=f'Cell Type {cell_type}',
                alpha=0.6,
                s=5
            )
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title('Spatial Cell Type Distribution')
        ax.legend()
        
        # Plot 2: Interacting vs non-interacting
        ax = axes[0, 1]
        for status in [True, False]:
            mask = self.adata.obs['is_interacting'] == status
            label = 'Interacting' if status else 'Non-interacting'
            ax.scatter(
                self.adata.obs.loc[mask, 'x'],
                self.adata.obs.loc[mask, 'y'],
                label=label,
                alpha=0.6,
                s=5
            )
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title('Interacting Cell States')
        ax.legend()
        
        # Plot 3: Batch distribution
        ax = axes[1, 0]
        for batch in self.batches:
            mask = self.adata.obs['batch'] == batch
            ax.scatter(
                self.adata.obs.loc[mask, 'x'],
                self.adata.obs.loc[mask, 'y'],
                label=batch,
                alpha=0.6,
                s=5
            )
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title('Batch Distribution')
        ax.legend()
        
        # Plot 4: Train/test split
        ax = axes[1, 1]
        for split in ['train', 'test']:
            mask = self.adata.obs['split'] == split
            ax.scatter(
                self.adata.obs.loc[mask, 'x'],
                self.adata.obs.loc[mask, 'y'],
                label=split,
                alpha=0.6,
                s=5
            )
        ax.axvline(x=900, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=1100, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title('Train/Test Split')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        
        return self

def generate_ba_amici_benchmark(
    adata_path="pbmc_data/pbmc_multi_batch.h5ad",
    n_replicates=10,
    output_dir="ba_amici_benchmark"
):
    """
    Generate multiple replicates of semi-synthetic data for BA-AMICI
    
    Parameters:
    -----------
    adata_path : str
        Path to preprocessed multi-batch PBMC data
    n_replicates : int
        Number of technical replicates to generate
    output_dir : str
        Directory to save outputs
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print("BA-AMICI Semi-Synthetic Benchmark Generation")
    print(f"Generating {n_replicates} replicates")
    print(f"{'='*60}")
    
    # Load preprocessed data
    print(f"\nLoading data from {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    
    # Generate replicates
    for rep in range(n_replicates):
        print(f"\n{'='*60}")
        print(f"Generating Replicate {rep + 1}/{n_replicates}")
        print(f"{'='*60}")
        
        # Create generator with different seed for each replicate
        generator = BASemiSyntheticGenerator(
            adata.copy(),
            n_cell_types=3,
            random_state=42 + rep
        )
        
        # Run generation pipeline
        generator.create_cell_type_clusters()
        generator.define_ground_truth_interactions()
        generator.generate_spatial_coordinates()
        generator.assign_interacting_phenotypes()
        generator.introduce_batch_effects(batch_effect_strength=0.5)
        generator.split_train_test()
        
        # Save
        rep_path = output_path / f"replicate_{rep:02d}.h5ad"
        generator.save(rep_path)
        
        # Visualize first replicate
        if rep == 0:
            viz_path = output_path / "replicate_00_visualization.png"
            generator.visualize(viz_path)
    
    print(f"\n{'='*60}")
    print("Benchmark Generation Complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")
    print(f"Generated {n_replicates} replicates")
    print("\nEach replicate contains:")
    print("  - Multiple batches with natural technical variation")
    print("  - Ground truth cell-cell interactions")
    print("  - Spatial organization")
    print("  - Train/test split")

if __name__ == "__main__":
    # Generate 10 replicates for BA-AMICI benchmark
    generate_ba_amici_benchmark(
        adata_path="pbmc_data/pbmc_multi_batch.h5ad",
        n_replicates=10,
        output_dir="ba_amici_benchmark"
    )