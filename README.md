# BA-AMICI: Batch-Aware Attention Mechanism Interpretation of Cell-cell Interactions

Batch-robust cell-cell interaction inference from spatial transcriptomics data.

## Overview

BA-AMICI extends the [AMICI framework](https://github.com/azizilab/amici) to handle batch effects across tissue replicates and experimental conditions. It introduces two key mechanisms:

1. **Batch-Conditioned Cross-Attention:** Learnable batch embeddings integrated into Q, K, V projections
2. **Adversarial Regularization:** Gradient reversal layer enforcing batch-invariant gene expression residuals

## Installation

You need to have Python 3.10 or newer installed on your system.

1) Clone the repository and install dependencies:

```bash
git clone https://github.com/[your-repo]/ba-amici.git
cd ba-amici
pip install -e .
```

Or install with uv:

```bash
uv pip install -e .
```

2) Required dependencies:

```
torch>=2.0
scvi-tools>=1.0
scanpy>=1.9
anndata>=0.9
pytorch-lightning>=2.0
numpy
pandas
```

## Quick Start

```python
import anndata
from amici import AMICI

# Load spatial transcriptomics data with batch information
adata = anndata.read_h5ad("./spatial_data.h5ad")

# Setup data with batch key
AMICI.setup_anndata(
    adata,
    labels_key="cell_type",
    batch_key="batch",           # Required for BA-AMICI
    coord_obsm_key="spatial",
    n_neighbors=50
)

# Create BA-AMICI model (batch-aware mode)
model = AMICI(
    adata,
    use_batch_aware=True,        # Enable batch-conditioned attention
    use_adversarial=True,        # Enable adversarial regularization
    lambda_adv=0.1,              # Adversarial loss weight
)

# Train
model.train(
    max_epochs=100,
    early_stopping=True,
    early_stopping_patience=10
)

# Get attention patterns
attention_patterns = model.get_attention_patterns()
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_batch_aware` | Enable batch-conditioned cross-attention | `False` |
| `use_adversarial` | Enable adversarial batch regularization | `False` |
| `lambda_adv` | Weight for adversarial loss | `0.1` |
| `n_neighbors` | Number of spatial neighbors per cell | `50` |
| `n_heads` | Number of attention heads | `4` |
| `hidden_dim` | Hidden dimension for embeddings | `256` |

## Model Architecture

### Batch-Conditioned Attention

Standard AMICI computes attention as:
```
Q = W_Q · h_recv
K = W_K · h_send
```

BA-AMICI adds batch conditioning:
```
Q = W_Q · h_recv + U_Q · E_batch_recv
K = W_K · h_send + U_K · E_batch_send
V = W_V · h_send + U_V · E_batch_send
```

Where `E_batch` is a learnable embedding for each batch.

### Adversarial Regularization

A discriminator attempts to predict batch from gene expression residuals:
```
D(Δ) → batch_prediction
```

The Gradient Reversal Layer (GRL) reverses gradients during backpropagation, training the encoder to produce batch-invariant residuals.

## Project Structure

```
ba-amici/
├── src/amici/
│   ├── _model.py              # AMICI model class
│   ├── _module.py             # PyTorch module with attention
│   ├── batch_attention.py     # BatchAwareCrossAttention
│   └── adversarial.py         # GRL and BatchDiscriminator
├── benchmarks/
    ├── ba_amici_benchmark/
        ├── ba_amici_benchmark_pipeline.py
        ├── data_generation/
        │   └── semisynthetic_batch_generator.py
        └── evaluation/
            └── interaction_consistency_evaluator.py
    
```

## Benchmarking

### Semi-Synthetic Data Generation

Generate benchmark data with known ground-truth interactions and batch effects:

```python
from benchmarks.data_generation import SemiBatchGenerator

generator = SemiBatchGenerator(
    source_adata=pbmc_adata,
    n_batches=3,
    n_replicates=10
)

# Generate replicates with batch effects
generator.generate_all_replicates(
    cells_per_replicate=20000,
    library_size_std=0.5,
    dropout_increase=0.15
)
```

### Evaluation

Compare BA-AMICI vs baseline AMICI on cross-replicate consistency:

```python
from benchmarks.evaluation import InteractionConsistencyEvaluator

evaluator = InteractionConsistencyEvaluator()

# Compute metrics
results = evaluator.evaluate(
    models={"baseline": baseline_model, "ba_amici": ba_amici_model},
    replicates=replicate_adatas
)

print(f"Interaction Jaccard: {results['jaccard']}")
print(f"Matrix Correlation: {results['correlation']}")
```

## Comparison: AMICI vs BA-AMICI

| Feature | AMICI | BA-AMICI |
|---------|-------|----------|
| Single-batch data | ✓ | ✓ |
| Multi-batch data | Limited | ✓ |
| Batch embeddings | ✗ | ✓ |
| Adversarial training | ✗ | ✓ |
| Cross-replicate consistency | Variable | Improved |

## Datasets

BA-AMICI has been tested on:

- **Semi-synthetic PBMC:** 68k PBMCs with simulated spatial coordinates and batch effects
- **Mouse Brain Visium:** 10x Visium CytAssist with 3 tissue replicates
- **Xenium Breast Cancer:** Two replicate slides (validation from original AMICI)

## References

This work builds upon:

```
@article{Hong2025.09.22.677860,
    title = {AMICI: Attention Mechanism Interpretation of Cell-cell Interactions},
    author = {Hong, Justin and Desai, Khushi and Nguyen, Tu Duyen and Nazaret, Achille 
              and Levy, Nathan and Ergen, Can and Plitas, George and Azizi, Elham},
    journal = {bioRxiv},
    year = {2025},
    doi = {10.1101/2025.09.22.677860}
}
```

Additional references:
- Ganin & Lempitsky (2015). Unsupervised Domain Adaptation by Backpropagation. *ICML*.
- Lopez et al. (2018). Deep generative modeling for single-cell transcriptomics. *Nature Methods*.

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
```python
# Reduce batch size
model.train(batch_size=64)
```

**2. Neighbor batch indices not found**
```python
# Ensure batch_key is specified in setup
AMICI.setup_anndata(adata, batch_key="batch", ...)
```

**3. Adversarial loss not decreasing**
```python
# Try adjusting lambda_adv
model = AMICI(adata, use_adversarial=True, lambda_adv=0.05)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

This project extends the AMICI framework developed by the Azizi Lab at Columbia University.
