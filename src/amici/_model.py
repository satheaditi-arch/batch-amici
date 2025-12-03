"""
BA-AMICI Model - FIXED VERSION

Key Fix: Registers neighbor batch indices in the data loader
so that BatchAwareCrossAttention can use actual sender batch IDs.
"""

import numpy as np
import pandas as pd
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
    """
    Fixed AMICI model with proper baseline vs BA-AMICI distinction.
    
    Key Changes:
    1. use_batch_aware flag controls attention type
    2. Neighbor batch indices are registered for proper batch conditioning
    3. Clear separation between baseline (no batch info) and BA-AMICI (with batch info)
    """
    
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
            ObsmField("nn_batch", nn_batch_key),
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
            loss_output = self.module(tensors, compute_loss=True)[1]
            total_loss += loss_output.loss.item() * tensors[REGISTRY_KEYS.X_KEY].shape[0]
            n_cells += tensors[REGISTRY_KEYS.X_KEY].shape[0]
        
        return {"reconstruction_loss": total_loss / n_cells}


# Update constants
if not hasattr(NN_REGISTRY_KEYS, 'NN_BATCH_KEY'):
    NN_REGISTRY_KEYS.NN_BATCH_KEY = "_nn_batch"