import scanpy as sc
import os

RAW_DIR = "data/xenium_mouse"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

def preprocess_rep(rep_name):
    path = os.path.join(RAW_DIR, rep_name)
    print(f"Loading {rep_name} from {path}")

    # Xenium output already has cell x gene matrix
    adata = sc.read_10x_h5(os.path.join(path, "cell_feature_matrix.h5"))

    # Cell filtering + normalization
    sc.pp.filter_cells(adata, min_counts=100)
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # Save
    out_path = os.path.join(OUT_DIR, f"xenium_{rep_name}_preprocessed.h5ad")
    adata.write(out_path)
    print("Saved:", out_path)

preprocess_rep("Rep1")
preprocess_rep("Rep2")
preprocess_rep("Rep3")