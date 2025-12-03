# benchmarks/xenium_mouse/build_xenium_mouse_graph.py

import scanpy as sc
import numpy as np
import os
import importlib.util

# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GRAPH_PATH = os.path.join(PROJECT_ROOT, "src", "amici", "graph.py")

spec = importlib.util.spec_from_file_location("graph", GRAPH_PATH)
graph_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_module)
build_joint_spatial_graph = graph_module.build_joint_spatial_graph

IN_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# =========================================================
# Load Preprocessed Xenium Replicates
# =========================================================

adata1 = sc.read_h5ad(f"{IN_DIR}/xenium_Rep1_preprocessed.h5ad")
adata2 = sc.read_h5ad(f"{IN_DIR}/xenium_Rep2_preprocessed.h5ad")
adata3 = sc.read_h5ad(f"{IN_DIR}/xenium_Rep3_preprocessed.h5ad")

adata1.obs["batch"] = "Rep1"
adata2.obs["batch"] = "Rep2"
adata3.obs["batch"] = "Rep3"

# =========================================================
# ✅ CONCATENATE FIRST (This creates `adata`)
# =========================================================

adata = sc.concat(
    [adata1, adata2, adata3],
    label="batch",
    keys=["Rep1", "Rep2", "Rep3"]
)

adata.obs_names_make_unique()

# =========================================================
# ✅ Xenium → Visium Spatial Fix (NOW `adata` EXISTS)
# =========================================================

if "spatial" not in adata.obsm:

    if "X_spatial" in adata.obsm:
        print("✅ Using Xenium obsm['X_spatial'] → mapping to obsm['spatial']")
        adata.obsm["spatial"] = adata.obsm["X_spatial"]

    elif {"x_centroid", "y_centroid"}.issubset(adata.obs.columns):
        print("✅ Using obs x/y centroids → mapping to obsm['spatial']")
        adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].values

    else:
        raise ValueError(
            "❌ Could not find Xenium spatial coordinates. "
            "Expected `obsm['X_spatial']` OR obs['x_centroid','y_centroid']"
        )

# =========================================================
# ✅ Build Joint Spatial Graph
# =========================================================

adata = build_joint_spatial_graph(adata, k=12)

# =========================================================
# ✅ Save Output
# =========================================================

out_path = f"{OUT_DIR}/xenium_merged_graph.h5ad"
adata.write(out_path)

print("✅ Xenium merged graph saved to:", out_path)
