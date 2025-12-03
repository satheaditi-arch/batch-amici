import scanpy as sc
import numpy as np
import os
import importlib.util

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GRAPH_PATH = os.path.join(PROJECT_ROOT, "src", "amici", "graph.py")

spec = importlib.util.spec_from_file_location("graph", GRAPH_PATH)
graph_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_module)
build_joint_spatial_graph = graph_module.build_joint_spatial_graph

IN_DIR = "data/processed"
OUT_DIR = "data/processed"

adata1 = sc.read_h5ad(f"{IN_DIR}/xenium_Rep1_preprocessed.h5ad")
adata2 = sc.read_h5ad(f"{IN_DIR}/xenium_Rep2_preprocessed.h5ad")
adata3 = sc.read_h5ad(f"{IN_DIR}/xenium_Rep3_preprocessed.h5ad")

adata1.obs["batch"] = "Rep1"
adata2.obs["batch"] = "Rep2"
adata3.obs["batch"] = "Rep3"

adata = sc.concat([adata1, adata2, adata3], label="batch", keys=["Rep1", "Rep2", "Rep3"])

adata = build_joint_spatial_graph(adata, k=12)

adata.write(f"{OUT_DIR}/xenium_merged_graph.h5ad")
print("Xenium merged graph saved.")
