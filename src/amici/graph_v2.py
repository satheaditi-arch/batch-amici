# src/amici/tools/graph.py

import numpy as np
from sklearn.neighbors import NearestNeighbors

<<<<<<< HEAD
def build_joint_spatial_graph(adata, k=12):
    coords = adata.obsm["spatial"].copy()
    batch = adata.obs["batch"].values

    coords_norm = coords.copy()
    for b in np.unique(batch):
        idx = batch == b
        sub = coords[idx]
        sub = (sub - sub.mean(0)) / (sub.std(0) + 1e-6)
        coords_norm[idx] = sub

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords_norm)
    distances, indices = nbrs.kneighbors(coords_norm)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # Enforce cross-batch neighbors
    for i in range(adata.n_obs):
        b_i = batch[i]
        neigh_batches = batch[indices[i]]
        if np.all(neigh_batches == b_i):
            other_idx = np.where(batch != b_i)[0]
            if len(other_idx) > 0:
                dists = np.linalg.norm(
                    coords_norm[other_idx] - coords_norm[i], axis=1
                )
                j = other_idx[np.argmin(dists)]
                indices[i, -1] = j
                distances[i, -1] = dists.min()

    adata.obsm["spatial_neighbors"] = indices
    adata.obsm["spatial_distances"] = distances
=======
import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_joint_spatial_graph(adata, k=12):
    """
    Build a joint kNN spatial graph using only numeric spatial coordinates.
    Safe for multi-batch mouse Visium data.
    """

    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] not found — Visium coordinates missing.")

    #  ONLY use numeric spatial coordinates (x, y)
    coords = np.asarray(adata.obsm["spatial"]).astype(float)

    print("Spatial coord shape:", coords.shape)
    print("Spatial coord dtype:", coords.dtype)

    # Normalize safely (now guaranteed numeric)
    coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-6)

    # Build kNN graph
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    # Store graph in AnnData (skip self-edge at index 0)
    adata.uns["spatial_neighbors"] = {
        "indices": indices[:, 1:],
        "distances": distances[:, 1:],
    }

    print("Joint spatial graph built with k =", k)

>>>>>>> 3766536c57ad21734b5c6efedd365083bc755118
    return adata
