import scanpy as sc
from amici import AMICI
import os

DATA_PATH = "data/processed/xenium_merged_graph.h5ad"
MODEL_PATH = "results/xenium/baseline/baseline_xenium_amici"
FIG_DIR = "results/xenium/baseline/figures"
os.makedirs(FIG_DIR, exist_ok=True)

adata = sc.read_h5ad(DATA_PATH)
model = AMICI.load(MODEL_PATH, adata=adata)

latent = model.get_latent_representation(adata)
adata.obsm["X_amici_base"] = latent

sc.pp.neighbors(adata, use_rep="X_amici_base")
sc.tl.umap(adata)

sc.pl.umap(adata, color="batch", save="_baseline_batch.png", show=False)
print("Baseline Xenium evaluation complete.")