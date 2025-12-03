import scanpy as sc
import os


import shutil


# Always resolve paths relative to THIS FILE
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "mouse_visium")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

print(">>> Script location:", HERE)
print(">>> Project root:", PROJECT_ROOT)
print(">>> Data dir:", DATA_DIR)

rep1_path = os.path.join(DATA_DIR, "Replicate1")
rep2_path = os.path.join(DATA_DIR, "Replicate2")

print(">>> Rep1 folder exists:", os.path.isdir(rep1_path))
print(">>> Rep2 folder exists:", os.path.isdir(rep2_path))

print(">>> Rep1 contents:", os.listdir(rep1_path))

h5_files = [f for f in os.listdir(rep1_path) if f.endswith(".h5")]
print(">>> Rep1 h5 files:", h5_files)

if len(h5_files) == 0:
    raise RuntimeError("NO .h5 FILE FOUND IN Replicate1")

def preprocess_visium(path):
    print(f"\nLoading Visium data from: {path}")
    print("Contents:", os.listdir(path))
    spatial_dir = os.path.join(path, "spatial")

    tp_csv = os.path.join(spatial_dir, "tissue_positions.csv")
    tp_list = os.path.join(spatial_dir, "tissue_positions_list.csv")

    if os.path.exists(tp_csv) and not os.path.exists(tp_list):
        print(">>> Creating compatibility copy: tissue_positions_list.csv")
        shutil.copy(tp_csv, tp_list)


    h5_files = [f for f in os.listdir(path) if f.endswith(".h5")]
    if len(h5_files) == 0:
        raise RuntimeError(f"No .h5 file found in {path}")

    h5_file = h5_files[0]
    print("Using H5 file:", h5_file)

    adata = sc.read_visium(
        path,
        count_file=h5_file,
        load_images=True,
    )

    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)

    return adata


print("\nPreprocessing Replicate 1...")
adata1 = preprocess_visium(rep1_path)
adata1.write(os.path.join(OUT_DIR, "mouse_rep1_preprocessed.h5ad"))

print("\nPreprocessing Replicate 2...")
adata2 = preprocess_visium(rep2_path)
adata2.write(os.path.join(OUT_DIR, "mouse_rep2_preprocessed.h5ad"))

print("\nSaved preprocessed mouse replicates successfully.")
