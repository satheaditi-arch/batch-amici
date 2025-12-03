import scanpy as sc
from amici import AMICI
import pytorch_lightning as pl
import os

DATA_PATH = "data/processed/xenium_merged_graph.h5ad"
OUT_DIR = "results/xenium/baseline"
os.makedirs(OUT_DIR, exist_ok=True)

adata = sc.read_h5ad(DATA_PATH)
pl.seed_everything(0)

AMICI.setup_anndata(
    adata,
    labels_key="batch",
    coord_obsm_key="spatial",
    n_neighbors=12,
)

model = AMICI(
    adata,
    n_heads=4,
    value_l1_penalty_coef=0.0,
)

model.train(
    max_epochs=150,
    batch_size=128,
    early_stopping=True,
)

model.save(f"{OUT_DIR}/baseline_xenium_amici", overwrite=True)
print("Baseline Xenium AMICI trained.")
