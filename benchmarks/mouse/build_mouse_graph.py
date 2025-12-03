# benchmarks/mouse/02_build_mouse_graph.py

import scanpy as sc
import os

import sys
import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
GRAPH_PATH = os.path.join(PROJECT_ROOT, "src", "amici", "graph.py")

spec = importlib.util.spec_from_file_location("graph", GRAPH_PATH)
graph_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_module)

build_joint_spatial_graph = graph_module.build_joint_spatial_graph

IN_DIR = "data/processed"
OUT_DIR = "data/processed"

adata1 = sc.read_h5ad(f"{IN_DIR}/mouse_rep1_preprocessed.h5ad")
adata2 = sc.read_h5ad(f"{IN_DIR}/mouse_rep2_preprocessed.h5ad")

adata1.obs["batch"] = "Rep1"
adata2.obs["batch"] = "Rep2"

adata = adata1.concatenate(
    adata2,
    batch_key="batch",
    batch_categories=["Rep1", "Rep2"]
)

adata = build_joint_spatial_graph(adata, k=12)

adata.write(f"{OUT_DIR}/mouse_merged_graph.h5ad")
print(" Saved merged mouse graph.")
