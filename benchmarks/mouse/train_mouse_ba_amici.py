# benchmarks/mouse/04_train_mouse_ba_amici.py

import scanpy as sc
from amici import AMICI
from amici.callbacks.lambda_adv_warmup import LambdaAdvWarmupCallback
import os

DATA_PATH = "data/processed/mouse_merged_graph.h5ad"
OUT_DIR = "results/mouse/ba_amici"
os.makedirs(OUT_DIR, exist_ok=True)

adata = sc.read_h5ad(DATA_PATH)

AMICI.setup_anndata(
    adata,
    labels_key= "batch",
    coord_obsm_key="spatial",
    n_neighbors=12,
)

model = AMICI(
    adata,
    n_heads=4,
    value_l1_penalty_coef=0.0,
)

lambda_cb = LambdaAdvWarmupCallback(
    warmup_epochs=20,
    max_val=0.2,
)

model.train(
    max_epochs=400,
    batch_size=128,
    early_stopping=True,
    callbacks=[lambda_cb],
)

model.save(f"{OUT_DIR}/model.pt", overwrite=True)

print(" BA-AMICI trained.")

print("BA-AMICI trained.")

