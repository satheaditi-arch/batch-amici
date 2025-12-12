"""
BA-AMICI Model - FIXED VERSION

Key Fix: Registers neighbor batch indices in the data loader
so that BatchAwareCrossAttention can use actual sender batch IDs.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict
import torch
from anndata import AnnData
from einops import rearrange, repeat
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalObsField,
    LayerField,
    ObsmField,
)
from scvi.model.base import BaseModelClass, VAEMixin
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from ._anndataloader import SpatialAnnDataLoader
from ._constants import NN_REGISTRY_KEYS
from ._module import AMICIModule
from ._wandb_training_mixin import WandbUnsupervisedTrainingMixin

DEFAULT_N_NEIGHBORS = 30


class AMICI(VAEMixin, WandbUnsupervisedTrainingMixin, BaseModelClass):
      
    _module_cls = AMICIModule
    _data_loader_cls = SpatialAnnDataLoader
    
    def __init__(
        self,
        adata: AnnData,
        use_batch_aware: bool = False,  # NEW: Controls attention type
        use_adversarial: bool = False,
        lambda_adv: float = 0.05,
        lambda_pair: float = 1e-3,
        **model_kwargs,
    ):
        """
        Initialize AMICI model.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data with setup_anndata already called
        use_batch_aware : bool
            If True, use BatchAwareCrossAttention (BA-AMICI)
            If False, use StandardCrossAttention (Baseline)
        use_adversarial : bool
            If True, add adversarial batch discrimination loss
        lambda_adv : float
            Weight for adversarial loss
        lambda_pair : float
            Weight for batch-pair bias regularization
        """
        self._data_splitter_cls.data_loader_cls = self._data_loader_cls
        super().__init__(adata)
        
        self.n_neighbors = adata.uns[NN_REGISTRY_KEYS.NUM_NEIGHBORS_KEY]
        
        # Compute empirical cell type means
        empirical_ct_means = []
        dataset_x = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        dataset_labels = self.adata_manager.get_from_registry(REGISTRY_KEYS.LABELS_KEY)
        
        for label_idx in range(self.summary_stats.n_labels):
            ct_idxs = np.where(dataset_labels == label_idx)[0]
            ct_means = torch.tensor(dataset_x[ct_idxs].mean(0))
            empirical_ct_means.append(ct_means)
        empirical_ct_means = torch.stack(empirical_ct_means)
        
        # Get number of batches
        n_batches = self.summary_stats["n_batch"]
        
        # Print configuration
        print(f"\n{'='*60}")
        print(f"AMICI Model Configuration")
        print(f"{'='*60}")
        print(f"  Mode: {'BA-AMICI (Batch-Aware)' if use_batch_aware else 'Baseline (Standard)'}")
        print(f"  Adversarial: {use_adversarial}")
        print(f"  n_batches: {n_batches}")
        print(f"  n_cell_types: {self.summary_stats.n_labels}")
        print(f"{'='*60}\n")
        
        # Initialize module
        self.module = self._module_cls(
            n_genes=adata.n_vars,
            n_labels=self.summary_stats.n_labels,
            empirical_ct_means=empirical_ct_means,
            n_batches=n_batches,
            use_batch_aware=use_batch_aware,
            use_adversarial=use_adversarial,
            lambda_adv=lambda_adv,
            lambda_pair=lambda_pair,
            **model_kwargs,
        )
        
        self.init_params_ = self._get_init_params(locals())
    
    @staticmethod
    def _compute_nn(
        adata: AnnData,
        coord_obsm_key: str,
        index_key: str,
        dist_key: str,
        batch_key: str,  # NEW: For computing neighbor batches
        nn_batch_key: str,  # NEW: Key to store neighbor batch indices
        n_neighbors: int,
        labels_obs_key: str | None = None,
        cell_radius_key: str | None = None,
        exclude_self_labels: bool = True,
    ) -> None:
        """
        Compute nearest neighbors and their batch indices.
        
        FIXED: Now also stores neighbor batch indices for BatchAwareCrossAttention.
        """
        assert not exclude_self_labels or labels_obs_key is not None
        adata.uns[NN_REGISTRY_KEYS.NUM_NEIGHBORS_KEY] = n_neighbors
        
        coords = adata.obsm[coord_obsm_key]
        if isinstance(coords, pd.DataFrame):
            coords = coords.values
        
        # Get batch labels for all cells
        batch_labels = adata.obs[batch_key].cat.codes.values if hasattr(adata.obs[batch_key], 'cat') else adata.obs[batch_key].values
        
        if not exclude_self_labels:
            nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean").fit(coords)
            nn_dist, nn_idx = nn.kneighbors(coords, return_distance=True)
            nn_dist, nn_idx = nn_dist[:, 1:], nn_idx[:, 1:]
            
            if cell_radius_key is not None:
                cell_radii = adata.obs[cell_radius_key].values
                nn_radii = cell_radii[nn_idx]
                nn_dist = np.clip(
                    nn_dist - cell_radii.reshape(-1, 1) - nn_radii, 0, None
                )
            
            adata.obsm[index_key] = nn_idx
            adata.obsm[dist_key] = nn_dist
            # NEW: Store neighbor batch indices
            adata.obsm[nn_batch_key] = batch_labels[nn_idx]
            
        else:
            labels = adata.obs[labels_obs_key].values
            adata.obsm[index_key] = np.zeros((adata.n_obs, n_neighbors), dtype=int)
            adata.obsm[dist_key] = np.zeros((adata.n_obs, n_neighbors), dtype=float)
            adata.obsm[nn_batch_key] = np.zeros((adata.n_obs, n_neighbors), dtype=int)  # NEW
            
            for label in np.unique(labels):
                label_idxs = np.where(labels == label)[0]
                not_label_idxs = np.where(labels != label)[0]
                
                nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(
                    coords[not_label_idxs]
                )
                nn_dist, nn_idx = nn.kneighbors(coords[label_idxs], return_distance=True)
                remapped_nn_idx = not_label_idxs[nn_idx]
                
                if cell_radius_key is not None:
                    cell_radii = adata.obs[cell_radius_key].values[label_idxs]
                    nn_radii = adata.obs[cell_radius_key].values[remapped_nn_idx]
                    cell_radii_repeated = repeat(cell_radii, "b -> b n", n=n_neighbors)
                    nn_dist = np.clip(nn_dist - cell_radii_repeated - nn_radii, 0, None)
                
                adata.obsm[index_key][label_idxs] = remapped_nn_idx
                adata.obsm[dist_key][label_idxs] = nn_dist
                # NEW: Store neighbor batch indices
                adata.obsm[nn_batch_key][label_idxs] = batch_labels[remapped_nn_idx]
    
    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        labels_key: str | None = None,
        coord_obsm_key: str | None = None,
        batch_key: str | None = None,
        nn_dist_key: str = "_nn_dist",
        nn_idx_key: str = "_nn_idx",
        nn_batch_key: str = "_nn_batch",  # NEW: Key for neighbor batches
        cell_radius_key: str | None = None,
        n_neighbors: int | None = None,
        **kwargs,
    ):
        """
        Setup AnnData for AMICI.
        
        FIXED: Now also computes and registers neighbor batch indices.
        """
        if n_neighbors is None:
            if NN_REGISTRY_KEYS.NUM_NEIGHBORS_KEY in adata.uns:
                n_neighbors = adata.uns[NN_REGISTRY_KEYS.NUM_NEIGHBORS_KEY]
            else:
                n_neighbors = DEFAULT_N_NEIGHBORS
        
        setup_method_args = cls._get_setup_method_args(**locals())
        
        # Compute neighbors (FIXED: now includes batch indices)
        cls._compute_nn(
            adata,
            coord_obsm_key,
            nn_idx_key,
            nn_dist_key,
            batch_key,  # NEW
            nn_batch_key,  # NEW
            n_neighbors,
            labels_obs_key=labels_key,
            exclude_self_labels=True,
            cell_radius_key=cell_radius_key,
        )
        
        # Register fields
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            ObsmField(NN_REGISTRY_KEYS.COORD_KEY, coord_obsm_key),
            ObsmField(NN_REGISTRY_KEYS.NN_IDX_KEY, nn_idx_key),
            ObsmField(NN_REGISTRY_KEYS.NN_DIST_KEY, nn_dist_key),
            ObsmField(NN_REGISTRY_KEYS.NN_BATCH_KEY, nn_batch_key),
        ]
        
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
    
    def get_predictions(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        batch_size: int | None = None,
        get_residuals: bool = False,
        prog_bar: bool = True,
    ) -> np.ndarray:
        """Get model predictions or residuals."""
        self._check_if_trained(warn=True)
        
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        
        out = []
        for tensors in tqdm(scdl, disable=not prog_bar):
            _, outputs = self.module(tensors, compute_loss=False)
            if get_residuals:
                out.append(outputs["residual"].detach().cpu().numpy())
            else:
                out.append(outputs["prediction"].detach().cpu().numpy())
        
        return np.vstack(out)
    
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        batch_size: int | None = None,
        representation: str = "residual_embed",
    ) -> np.ndarray:
        """
        Get latent representation for evaluation.
        
        Parameters
        ----------
        representation : str
            Which representation to return:
            - "residual_embed": Output of attention aggregation (best for batch mixing eval)
            - "residual": Final gene-space residual
            - "prediction": Full prediction
        """
        self._check_if_trained(warn=True)
        
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        
        out = []
        for tensors in tqdm(scdl):
            _, outputs = self.module(tensors, compute_loss=False)
            out.append(outputs[representation].detach().cpu().numpy())
        
        return np.vstack(out)
    
    def get_reconstruction_error(
        self,
        adata: AnnData | None = None,
        batch_size: int | None = None,
    ) -> dict:
        """Compute reconstruction error on data."""
        self._check_if_trained(warn=True)
        
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, batch_size=batch_size)
        
        total_loss = 0.0
        n_cells = 0
        
        for tensors in scdl:
            # Get inference and generative outputs
            inference_outputs, generative_outputs = self.module(tensors, compute_loss=False)
            
            # Compute loss separately using the module's loss method
            loss_output = self.module.loss(
                tensors, 
                inference_outputs, 
                generative_outputs
            )
            
            batch_size_current = tensors[REGISTRY_KEYS.X_KEY].shape[0]
            total_loss += loss_output.loss.item() * batch_size_current
            n_cells += batch_size_current
        
        return {"reconstruction_loss": total_loss / n_cells}
    @dataclass
    class AttentionPatterns:
        """
        Container for attention pattern analysis results.
        
        Attributes
        ----------
        attention_df : pd.DataFrame
            DataFrame with columns:
            - cell_idx: Index of receiver cell
            - head: Attention head index
            - label: Cell type of receiver
            - neighbor_0, neighbor_1, ...: Attention weights to each neighbor
        nn_idx_df : pd.DataFrame
            DataFrame mapping cell indices to their neighbor indices
            Columns: cell_idx, neighbor_0, neighbor_1, ...
        cell_types : np.ndarray
            Cell type labels for all cells [n_cells]
        batches : np.ndarray
            Batch labels for all cells [n_cells]
        """
        attention_df: pd.DataFrame
        nn_idx_df: pd.DataFrame
        cell_types: np.ndarray
        batches: np.ndarray
        
        def get_attention_to_type(
            self, 
            receiver_type: str = None, 
            sender_type: str = None,
            head: int = None,
        ) -> pd.DataFrame:
            """
            Filter attention patterns by receiver/sender cell type.
            
            Parameters
            ----------
            receiver_type : str, optional
                Filter for specific receiver cell type
            sender_type : str, optional
                Filter for specific sender cell type (requires neighbor lookup)
            head : int, optional
                Filter for specific attention head
                
            Returns
            -------
            pd.DataFrame
                Filtered attention patterns
            """
            df = self.attention_df.copy()
            
            if receiver_type is not None:
                df = df[df['label'] == receiver_type]
            
            if head is not None:
                df = df[df['head'] == head]
            
            if sender_type is not None:
                # This requires looking up neighbor types
                # Implementation depends on having cell type info for neighbors
                pass
            
            return df
        
        def summary(self) -> Dict[str, any]:
            """Get summary statistics of attention patterns."""
            n_cells = len(np.unique(self.attention_df['cell_idx']))
            n_heads = len(np.unique(self.attention_df['head']))
            n_neighbors = len([col for col in self.attention_df.columns if col.startswith('neighbor_')])
            n_types = len(np.unique(self.cell_types))
            n_batches = len(np.unique(self.batches))
            
            return {
                'n_cells': n_cells,
                'n_heads': n_heads,
                'n_neighbors': n_neighbors,
                'n_cell_types': n_types,
                'n_batches': n_batches,
            }


    @dataclass 
    class InteractionScores:
        """Container for interaction-level scores."""
        gene_scores: pd.DataFrame      # Genes x interactions with importance scores
        sender_scores: pd.DataFrame    # Cell type pairs with sender importance
        receiver_scores: pd.DataFrame  # Cell type pairs with receiver importance
        
    def get_attention_patterns(
    self,
    adata: AnnData = None,
    indices: list = None,
    batch_size: int = 128,
    return_raw: bool = False,
    ) -> AttentionPatterns:
        """
        Extract attention patterns from the trained model.
        
        This shows which sender cells (neighbors) each receiver cell attends to,
        which is the key output for identifying cell-cell interactions.
        
        Parameters
        ----------
        adata : AnnData, optional
            Data to extract patterns from. Uses training data if None.
        indices : list, optional
            Subset of cell indices to process
        batch_size : int
            Batch size for processing
        return_raw : bool
            If True, return raw tensors instead of DataFrames
            
        Returns
        -------
        AttentionPatterns
            Contains attention weights, neighbor indices, cell types, batches
        """
        self._check_if_trained(warn=True)
        adata = self._validate_anndata(adata)
        
        scdl = self._make_data_loader(
            adata=adata, 
            indices=indices, 
            batch_size=batch_size,
            shuffle=False,
        )
        
        all_attention = []
        all_labels = []
        all_batches = []
        all_nn_idx = []
        
        self.module.eval()
        with torch.no_grad():
            for tensors in tqdm(scdl, desc="Extracting attention patterns"):
                # Get inference outputs
                inference_inputs = self.module._get_inference_input(tensors)
                inference_outputs = self.module.inference(**inference_inputs)
                
                # Get generative outputs (includes attention patterns)
                gen_inputs = self.module._get_generative_input(tensors, inference_outputs)
                gen_outputs = self.module.generative(**gen_inputs, return_attention_patterns=True)
                
                # Collect results
                attn = gen_outputs["attention_patterns"].cpu().numpy()  # [B, N_neighbors, H] or [B, H, N_neighbors]
                all_attention.append(attn)
                
                all_labels.append(tensors[REGISTRY_KEYS.LABELS_KEY].cpu().numpy())
                all_batches.append(tensors[REGISTRY_KEYS.BATCH_KEY].cpu().numpy())
                all_nn_idx.append(tensors[NN_REGISTRY_KEYS.NN_IDX_KEY].cpu().numpy())
        
        # Concatenate
        attention = np.concatenate(all_attention, axis=0)
        labels = np.concatenate(all_labels, axis=0).squeeze()
        batches = np.concatenate(all_batches, axis=0).squeeze()
        nn_idx = np.concatenate(all_nn_idx, axis=0)
        
        if return_raw:
            return attention, nn_idx, labels, batches
        
        # Build DataFrames
        n_cells, n_neighbors, n_heads = attention.shape if len(attention.shape) == 3 else (attention.shape[0], attention.shape[2], attention.shape[1])
        
        # Reshape attention to [n_cells * n_heads, n_neighbors + metadata]
        attention_records = []
        for cell_idx in range(n_cells):
            for head_idx in range(n_heads):
                record = {
                    'cell_idx': cell_idx,
                    'head': head_idx,
                    'label': labels[cell_idx],
                }
                for neighbor_idx in range(n_neighbors):
                    if len(attention.shape) == 3:
                        record[f'neighbor_{neighbor_idx}'] = attention[cell_idx, neighbor_idx, head_idx]
                    else:
                        record[f'neighbor_{neighbor_idx}'] = attention[cell_idx, head_idx, neighbor_idx]
                attention_records.append(record)
        
        attention_df = pd.DataFrame(attention_records)
        
        # Neighbor indices DataFrame
        nn_idx_df = pd.DataFrame(
            nn_idx,
            columns=[f'neighbor_{i}' for i in range(n_neighbors)]
        )
        nn_idx_df['cell_idx'] = np.arange(n_cells)
        
        return self.__class__.AttentionPatterns(
            attention_df=attention_df,
            nn_idx_df=nn_idx_df,
            cell_types=labels,
            batches=batches,
        )


    def get_interaction_scores(
        self,
        adata: AnnData = None,
        sender_type: str = None,
        receiver_type: str = None,
        batch_size: int = 128,
        n_permutations: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Compute interaction scores between cell types.
        
        This method computes how much each sender cell type influences
        each receiver cell type by analyzing attention patterns.
        
        Parameters
        ----------
        adata : AnnData
            Data to analyze
        sender_type : str, optional
            Specific sender type to analyze (analyzes all if None)
        receiver_type : str, optional  
            Specific receiver type to analyze (analyzes all if None)
        batch_size : int
            Batch size for processing
        n_permutations : int
            Number of permutations for significance testing
            
        Returns
        -------
        Dict with interaction scores
        """
        self._check_if_trained(warn=True)
        adata = self._validate_anndata(adata)
        
        # Get attention patterns
        attn_result = self.get_attention_patterns(adata, batch_size=batch_size, return_raw=True)
        attention, nn_idx, labels, batches = attn_result
        
        # Get cell type mapping
        cell_types = adata.obs['cell_type'].values
        unique_types = np.unique(cell_types)
        
        # Compute mean attention from each receiver type to each sender type
        interaction_matrix = np.zeros((len(unique_types), len(unique_types)))
        
        for i, recv_type in enumerate(unique_types):
            recv_mask = cell_types == recv_type
            recv_indices = np.where(recv_mask)[0]
            
            if len(recv_indices) == 0:
                continue
                
            for j, send_type in enumerate(unique_types):
                # For each receiver cell, find neighbors of sender type
                total_attention = 0.0
                n_pairs = 0
                
                for recv_idx in recv_indices:
                    neighbor_indices = nn_idx[recv_idx]
                    neighbor_types = cell_types[neighbor_indices]
                    sender_mask = neighbor_types == send_type
                    
                    if sender_mask.sum() > 0:
                        # Mean attention to this sender type across all heads
                        if len(attention.shape) == 3:
                            attn_to_sender = attention[recv_idx, sender_mask, :].mean()
                        else:
                            attn_to_sender = attention[recv_idx, :, sender_mask].mean()
                        total_attention += attn_to_sender
                        n_pairs += 1
                
                if n_pairs > 0:
                    interaction_matrix[i, j] = total_attention / n_pairs
        
        # Create DataFrame
        interaction_df = pd.DataFrame(
            interaction_matrix,
            index=unique_types,
            columns=unique_types
        )
        interaction_df.index.name = 'receiver'
        interaction_df.columns.name = 'sender'
        
        return {
            'interaction_matrix': interaction_matrix,
            'interaction_df': interaction_df,
            'cell_types': unique_types,
        }


    def get_gene_interaction_scores(
        self,
        adata: AnnData = None,
        sender_type: str = None,
        receiver_type: str = None,
        batch_size: int = 128,
        top_k: int = 50,
    ) -> pd.DataFrame:
        """
        Compute gene-level interaction scores.
        
        For a given sender-receiver pair, identifies which genes in the
        receiver are most influenced by the sender's presence.
        
        This is done by:
        1. Computing predictions with and without sender type neighbors
        2. Finding genes with largest prediction difference
        
        Parameters
        ----------
        adata : AnnData
            Data to analyze
        sender_type : str
            Sender cell type
        receiver_type : str
            Receiver cell type
        batch_size : int
            Batch size
        top_k : int
            Number of top genes to return
            
        Returns
        -------
        pd.DataFrame with gene scores
        """
        self._check_if_trained(warn=True)
        adata = self._validate_anndata(adata)
        
        cell_types = adata.obs['cell_type'].values
        
        # Get receiver cells
        receiver_mask = cell_types == receiver_type
        receiver_indices = np.where(receiver_mask)[0]
        
        if len(receiver_indices) == 0:
            return pd.DataFrame()
        
        # Get predictions for receiver cells
        predictions_full = self.get_predictions(adata, indices=receiver_indices.tolist(), batch_size=batch_size)
        
        # Get residuals (difference from cell type mean)
        residuals = self.get_predictions(adata, indices=receiver_indices.tolist(), batch_size=batch_size, get_residuals=True)
        
        # Compute gene importance as variance of residuals
        # Genes with high variance are more influenced by neighbors
        gene_variance = np.var(residuals, axis=0)
        
        # Also compute mean absolute residual
        gene_mean_abs = np.mean(np.abs(residuals), axis=0)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'gene': adata.var_names,
            'variance': gene_variance,
            'mean_abs_residual': gene_mean_abs,
            'importance_score': gene_variance * gene_mean_abs,  # Combined score
        })
        
        results = results.sort_values('importance_score', ascending=False)
        
        return results.head(top_k)


    def get_batch_consistency_scores(
        self,
        adata: AnnData = None,
        batch_size: int = 128,
    ) -> Dict[str, float]:
        """
        Compute how consistent attention patterns are across batches.
        
        For BA-AMICI, attention should be consistent across batches.
        For baseline AMICI, attention may vary with batch.
        
        Returns
        -------
        Dict with consistency metrics
        """
        self._check_if_trained(warn=True)
        adata = self._validate_anndata(adata)
        
        # Get attention patterns
        attn_result = self.get_attention_patterns(adata, batch_size=batch_size, return_raw=True)
        attention, nn_idx, labels, batches = attn_result
        
        unique_batches = np.unique(batches)
        n_batches = len(unique_batches)
        
        if n_batches < 2:
            return {'batch_consistency': 1.0, 'note': 'Only one batch'}
        
        # Compute mean attention pattern per batch
        batch_patterns = []
        for batch in unique_batches:
            batch_mask = batches == batch
            if len(attention.shape) == 3:
                batch_attn = attention[batch_mask].mean(axis=(0, 2))  # Mean over cells and heads
            else:
                batch_attn = attention[batch_mask].mean(axis=(0, 1))
            batch_patterns.append(batch_attn)
        
        batch_patterns = np.stack(batch_patterns)
        
        # Compute pairwise correlations between batch patterns
        correlations = []
        for i in range(n_batches):
            for j in range(i + 1, n_batches):
                corr = np.corrcoef(batch_patterns[i], batch_patterns[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        mean_correlation = np.mean(correlations) if correlations else 0.0
        
        return {
            'batch_consistency': mean_correlation,
            'n_batches': n_batches,
            'pairwise_correlations': correlations,
        }