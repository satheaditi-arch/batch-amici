import scanpy as sc
import numpy as np
from amici import AMICI
import os

DATA_PATH = "data/processed/xenium_merged_graph.h5ad"
MODEL_PATH = "results/xenium/ba_amici/model.pt"
FIG_DIR = "results/xenium/ba_amici/figures"
os.makedirs(FIG_DIR, exist_ok=True)

adata = sc.read_h5ad(DATA_PATH)
model = AMICI.load(MODEL_PATH, adata=adata)

loader = model._make_data_loader(adata, batch_size=256)
all_latent = []

for batch in loader:
    _, generative_out = model.module.forward(batch, compute_loss=False)
    latent = generative_out["residual_embed"].detach().cpu().numpy()
    all_latent.append(latent)

all_latent = np.vstack(all_latent)
adata.obsm["X_ba_amici"] = all_latent

sc.pp.neighbors(adata, use_rep="X_ba_amici")
sc.tl.umap(adata)

sc.pl.umap(adata, color="batch", save="_ba_batch.png", show=False)
print("BA-AMICI Xenium evaluation complete.")
