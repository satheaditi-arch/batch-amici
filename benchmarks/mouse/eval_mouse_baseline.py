# benchmarks/mouse/05_eval_mouse_baseline.py

import scanpy as sc
from amici import AMICI
import os
import pytorch_lightning as pl
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "mouse_merged_graph.h5ad")
MODEL_PATH = os.path.join(PROJECT_ROOT, "results", "mouse_baseline", "baseline_mouse_amici")
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "mouse_baseline", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

pl.seed_everything(0)
adata = sc.read_h5ad(DATA_PATH)
model = AMICI.load(MODEL_PATH, adata=adata)

loader = model._make_data_loader(adata, batch_size=256)
all_latent = []

for batch in loader:

    inference_out , generative_out = model.module.forward(batch, compute_loss = False)
    latent = generative_out["residual_embed"].detach().cpu().numpy()
    all_latent.append(latent)

all_latent = np.vstack(all_latent)
adata.obsm["X_amici_base"] = all_latent


sc.pp.neighbors(adata, use_rep="X_amici_base")
sc.tl.umap(adata)

sc.pl.umap(adata, color="batch", save="_baseline_batch.png",
           show=False)
print(" Baseline evaluation plots saved.")
