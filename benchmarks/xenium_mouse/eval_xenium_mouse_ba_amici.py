# benchmarks/xenium_mouse/eval_xenium_ba_amici.py

import os
import numpy as np
import torch
import scanpy as sc
import pytorch_lightning as pl

from amici import AMICI

# -------------------------------------------------
# Paths
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "xenium_merged_graph.h5ad")
MODEL_DIR = os.path.join(PROJECT_ROOT, "results", "xenium", "ba_amici")
STATE_PATH = os.path.join(MODEL_DIR, "model.pt")

FIG_DIR = os.path.join(MODEL_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# -------------------------------------------------
# Load data
# -------------------------------------------------
pl.seed_everything(0)
adata = sc.read_h5ad(DATA_PATH)

print("Xenium data loaded:", adata)
print(" Batch counts:\n", adata.obs["batch"].value_counts())

# -------------------------------------------------
# Rebuild AMICI architecture (same as training)
# -------------------------------------------------
# IMPORTANT: we DO NOT use AMICI.load() here.
# We reconstruct the model and then load the state_dict by hand.

AMICI.setup_anndata(
    adata,
    labels_key="batch",
    coord_obsm_key="spatial",  # or "spatial" if that's what you used
    n_neighbors=12,
)

model = AMICI(
    adata,
    n_heads=1,
    n_head_size=256,
    n_query_dim = 32,   
    n_query_embed_hidden= 512,
    value_l1_penalty_coef=0.0,
    # we DON'T pass use_adversarial here; we just load weights
)

print(" Fresh BA-AMICI instance created.")
print(" Loading state_dict from:", STATE_PATH)

state = torch.load(STATE_PATH, map_location="cpu", weights_only=False)


# scvi's save format sometimes nests the true state_dict
if isinstance(state, dict) and "state_dict" in state:
    state_dict = state["state_dict"]
elif isinstance(state, dict) and "model_state_dict" in state:
    state_dict = state["model_state_dict"]
else:
    state_dict = state

missing, unexpected = model.module.load_state_dict(state_dict, strict=False)
print(" Loaded weights.")
if missing:
    print("Missing keys (ignored):", missing)
if unexpected:
    print(" Unexpected keys (ignored):", unexpected)

# -------------------------------------------------
# Extract latent representation from residual embeddings
# -------------------------------------------------
loader = model._make_data_loader(adata, batch_size=256)
all_latent = []

for batch in loader:
    _, gen_out = model.module.forward(batch, compute_loss=False)
    latent = gen_out["residual_embed"].detach().cpu().numpy()
    all_latent.append(latent)

all_latent = np.vstack(all_latent)
adata.obsm["X_ba_amici"] = all_latent

print("Latent BA-AMICI representation stored in adata.obsm['X_ba_amici'].")

# -------------------------------------------------
# UMAP + plotting
# -------------------------------------------------
sc.pp.neighbors(adata, use_rep="X_ba_amici")
sc.tl.umap(adata)

sc.pl.umap(
    adata,
    color="batch",
    save="_ba_batch.png",
    show=False,
)

print(" BA-AMICI Xenium evaluation complete. UMAP saved to:")
print("   results/xenium/ba_amici/figures/umap_ba_batch.png (or similar)")
