# benchmarks/xenium_mouse/train_xenium_ba_amici.py

import scanpy as sc
from amici import AMICI
from amici.callbacks.lambda_adv_warmup import LambdaAdvWarmupCallback
import os

DATA_PATH = "data/processed/xenium_merged_graph.h5ad"
OUT_DIR = "results/xenium/ba_amici"
os.makedirs(OUT_DIR, exist_ok=True)

adata = sc.read_h5ad(DATA_PATH)


AMICI.setup_anndata(
    adata,
    labels_key="batch",
    batch_key = "batch",
    coord_obsm_key="spatial",
    n_neighbors=12,
)

model = AMICI(
    adata,
    use_adversarial=True,
    n_heads=2,                 
)


lambda_cb = LambdaAdvWarmupCallback(
    warmup_epochs=10, 
    max_val=0.2,
)


model.train(
    max_epochs=120,          
    batch_size=256,          
    early_stopping=True,
    early_stopping_patience=10,
    callbacks=[lambda_cb],
)


model.save(f"{OUT_DIR}/model.pt", overwrite=True)
print("✅ BA-AMICI Xenium trained (fast config).")
