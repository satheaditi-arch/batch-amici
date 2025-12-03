import os
import scanpy as sc
import pytorch_lightning as pl

from amici import AMICI


# =========================================================

#  Paths

# Paths

# =========================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "mouse_merged_graph.h5ad")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "mouse_baseline")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================


# Load data

# =========================================================

adata = sc.read_h5ad(DATA_PATH)
pl.seed_everything(0)

print("Data loaded:", adata)

print(" Data loaded:", adata)

print("batch counts:\n", adata.obs["batch"].value_counts())

# =========================================================
#  Setup AMICI
# =========================================================

AMICI.setup_anndata(
    adata,
    labels_key="batch",
    coord_obsm_key="spatial",
    n_neighbors=12,
)

# =========================================================
# Train BASELINE AMICI
# =========================================================

print("\n Training BASELINE AMICI on Mouse Visium...\n")

model = AMICI(
    adata,
    n_heads=4,
    value_l1_penalty_coef=0.0,
)

model.train(
    max_epochs=120,
    batch_size=128,
    early_stopping=True,
    early_stopping_patience=15,
    check_val_every_n_epoch=1,
)

# =========================================================
#  Save model
# =========================================================

out_path = os.path.join(OUT_DIR, "baseline_mouse_amici")
model.save(out_path, overwrite=True)

print(" BASELINE TRAINING COMPLETE")

print("Saved to:", out_path)

print(" Saved to:", out_path)

