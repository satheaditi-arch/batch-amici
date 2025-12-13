# benchmarks/xenium_mouse/eval_xenium_mouse_baseline.py

import os
import numpy as np
import scanpy as sc
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from amici import AMICI
from scib.metrics import ilisi_graph, kBET

# -------------------------------------------------
# Paths
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "xenium_merged_graph.h5ad")
MODEL_DIR = os.path.join(PROJECT_ROOT, "results", "xenium", "baseline", "baseline_xenium_amici")
CKPT_PATH = os.path.join(MODEL_DIR, "model.pt")
FIG_DIR = os.path.join(MODEL_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

pl.seed_everything(0)

# -------------------------------------------------
# Load data
# -------------------------------------------------
adata = sc.read_h5ad(DATA_PATH)
print("Batch counts:")
print(adata.obs["batch"].value_counts())

# -------------------------------------------------
# 🔑 REGISTER ANNDATA (THIS MUST MATCH TRAINING)
# -------------------------------------------------
AMICI.setup_anndata(
    adata,
    batch_key ="batch", 
    labels_key="batch",
    coord_obsm_key="spatial",
    n_neighbors=12,
)

# -------------------------------------------------
# 🔑 REBUILD MODEL (MATCH TRAINING PARAMS)
# -------------------------------------------------
model = AMICI(
    adata,
    n_heads=2,                 # ← matches training
    value_l1_penalty_coef=0.0,
)

# -------------------------------------------------
# 🔑 LOAD WEIGHTS ONLY (BYPASS REGISTRY)
# -------------------------------------------------
print("Loading weights directly from checkpoint...")
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

state_dict = ckpt.get("state_dict", ckpt)
missing, unexpected = model.module.load_state_dict(state_dict, strict=False)

print("Loaded weights.")
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

model.module.eval()

# -------------------------------------------------
# Extract latent representation
# -------------------------------------------------
loader = model._make_data_loader(adata, batch_size=256)
latent = []

for i, batch in enumerate(loader):
    _, gen_out = model.module.forward(batch, compute_loss=False)
    latent.append(gen_out["residual_embed"].detach().cpu().numpy())

    if i % 100 == 0:
        print(f"Processed {i} batches")

adata.obsm["X_amici_base"] = np.vstack(latent)

# -------------------------------------------------
# UMAP
# -------------------------------------------------
sc.pp.neighbors(adata, use_rep="X_amici_base")
sc.tl.umap(adata)

sc.pl.umap(
    adata,
    color="batch",
    show=False,
    save="_baseline_batches.png",
)

# -------------------------------------------------
# Metrics
# -------------------------------------------------
ilisi = ilisi_graph(adata, batch_key="batch", type = "knn", use_rep="X_amici_base")
kbet = kBET(adata, batch_key="batch", embed="X_amici_base")

metrics = pd.DataFrame({
    "model": ["Baseline"],
    "iLISI": [np.mean(ilisi)],
    "kBET": [kbet],
})

metrics.to_csv(os.path.join(FIG_DIR, "baseline_metrics.csv"), index=False)

print(metrics)
print("✅ Baseline Xenium evaluation COMPLETE")
