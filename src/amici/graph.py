# src/amici/tools/graph.py

import numpy as np
from sklearn.neighbors import NearestNeighbors

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

    return adata
