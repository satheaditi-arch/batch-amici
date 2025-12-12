"""
BA-AMICI Diagnostic Script

This script checks:
1. Data integrity (replicates, ground truth)
2. Model loading and functionality
3. Prediction extraction capabilities
4. AMICI interpretation methods

Run this to identify where your pipeline is failing.
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from amici import AMICI
# Configuration - UPDATE THESE PATHS
DATA_DIR = "results/ba_amici_benchmark/data"
MODELS_DIR = "results/ba_amici_benchmark/models"
REPLICATE_IDX = 0  # Which replicate to test

print("="*70)
print("BA-AMICI DIAGNOSTIC SCRIPT")
print("="*70)

# =============================================================================
# 1. CHECK DATA
# =============================================================================
print("\n" + "="*70)
print("1. CHECKING DATA")
print("="*70)

rep_path = os.path.join(DATA_DIR, f"replicate_{REPLICATE_IDX:02d}.h5ad")
print(f"\nLoading: {rep_path}")

if not os.path.exists(rep_path):
    print(f"ERROR: File not found: {rep_path}")
    sys.exit(1)

adata = sc.read_h5ad(rep_path)
print(f"✓ Loaded successfully")
print(f"  Shape: {adata.shape}")
print(f"  obs columns: {list(adata.obs.columns)}")
print(f"  obsm keys: {list(adata.obsm.keys())}")
print(f"  uns keys: {list(adata.uns.keys())}")
print(f"  layers: {list(adata.layers.keys())}")

# Check required columns
required_obs = ['cell_type', 'batch', 'subtype']
for col in required_obs:
    if col in adata.obs.columns:
        print(f"  ✓ {col}: {adata.obs[col].nunique()} unique values")
        print(f"    Values: {adata.obs[col].unique().tolist()[:5]}...")
    else:
        print(f"  ✗ MISSING: {col}")

# Check spatial coordinates
if 'spatial' in adata.obsm:
    print(f"  ✓ spatial coordinates shape: {adata.obsm['spatial'].shape}")
else:
    print("  ✗ MISSING: spatial coordinates in obsm")

# Check for neighbor indices (needed for AMICI)
if '_nn_idx' in adata.obsm:
    print(f"  ✓ _nn_idx shape: {adata.obsm['_nn_idx'].shape}")
else:
    print("  ✗ MISSING: _nn_idx (neighbor indices) - AMICI.setup_anndata() needed")

if '_nn_dist' in adata.obsm:
    print(f"  ✓ _nn_dist shape: {adata.obsm['_nn_dist'].shape}")
else:
    print("  ✗ MISSING: _nn_dist (neighbor distances) - AMICI.setup_anndata() needed")

# Check expression data
print(f"\n  Expression matrix:")
print(f"    Type: {type(adata.X)}")
print(f"    Range: [{adata.X.min():.3f}, {adata.X.max():.3f}]")
print(f"    Mean: {adata.X.mean():.3f}")

# =============================================================================
# 2. SETUP AMICI ON DATA
# =============================================================================
print("\n" + "="*70)
print("2. SETTING UP AMICI ON DATA")
print("="*70)

try:
    from amici import AMICI
    print("✓ AMICI imported successfully")
except ImportError as e:
    print(f"✗ Failed to import AMICI: {e}")
    sys.exit(1)

try:
    AMICI.setup_anndata(
        adata,
        labels_key="cell_type",
        batch_key="batch",
        coord_obsm_key="spatial",
        n_neighbors=50,
    )
    print("✓ AMICI.setup_anndata() completed")
    print(f"  _nn_idx shape: {adata.obsm['_nn_idx'].shape}")
    print(f"  _nn_dist shape: {adata.obsm['_nn_dist'].shape}")
except Exception as e:
    print(f"✗ AMICI.setup_anndata() failed: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# 3. CHECK MODELS
# =============================================================================
print("\n" + "="*70)
print("3. CHECKING MODELS")
print("="*70)

baseline_path = os.path.join(MODELS_DIR, f"replicate_{REPLICATE_IDX:02d}", "baseline_amici")
ba_amici_path = os.path.join(MODELS_DIR, f"replicate_{REPLICATE_IDX:02d}", "ba_amici")

print(f"\nBaseline model path: {baseline_path}")
print(f"  Exists: {os.path.exists(baseline_path)}")

print(f"\nBA-AMICI model path: {ba_amici_path}")
print(f"  Exists: {os.path.exists(ba_amici_path)}")

# List model files
if os.path.exists(baseline_path):
    print(f"\n  Baseline model files:")
    for f in os.listdir(baseline_path):
        fpath = os.path.join(baseline_path, f)
        size = os.path.getsize(fpath) / 1024  # KB
        print(f"    {f}: {size:.1f} KB")

# =============================================================================
# 4. LOAD AND TEST MODELS
# =============================================================================
print("\n" + "="*70)
print("4. LOADING AND TESTING MODELS")
print("="*70)

baseline_model = None
ba_amici_model = None

# Load baseline
try:
    print("\nLoading baseline model...")
    baseline_model = AMICI.load(baseline_path, adata=adata)
    print("✓ Baseline model loaded")
    print(f"  Type: {type(baseline_model)}")
    print(f"  Module type: {type(baseline_model.module)}")
    print(f"  n_neighbors: {baseline_model.n_neighbors}")
except Exception as e:
    print(f"✗ Failed to load baseline model: {e}")
    import traceback
    traceback.print_exc()

# Load BA-AMICI
try:
    print("\nLoading BA-AMICI model...")
    ba_amici_model = AMICI.load(ba_amici_path, adata=adata)
    print("✓ BA-AMICI model loaded")
    print(f"  Type: {type(ba_amici_model)}")
    print(f"  Module type: {type(ba_amici_model.module)}")
except Exception as e:
    print(f"✗ Failed to load BA-AMICI model: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# 5. TEST ATTENTION PATTERN EXTRACTION
# =============================================================================
print("\n" + "="*70)
print("5. TESTING ATTENTION PATTERN EXTRACTION")
print("="*70)

if baseline_model is not None:
    try:
        print("\nExtracting attention patterns from baseline model...")
        attention_module = baseline_model.get_attention_patterns(
            adata=adata,
            batch_size=128,
        )
        print("✓ Attention patterns extracted")
        
        if hasattr(attention_module, '_attention_patterns_df'):
            attn_df = attention_module._attention_patterns_df
            print(f"  DataFrame shape: {attn_df.shape}")
            print(f"  Columns: {list(attn_df.columns)[:10]}...")
            print(f"  Sample values:")
            print(attn_df.head())
            
            # Check for non-zero attention
            neighbor_cols = [c for c in attn_df.columns if c.startswith('neighbor_')]
            if neighbor_cols:
                attn_values = attn_df[neighbor_cols].values
                print(f"\n  Attention statistics:")
                print(f"    Min: {attn_values.min():.6f}")
                print(f"    Max: {attn_values.max():.6f}")
                print(f"    Mean: {attn_values.mean():.6f}")
                print(f"    Non-zero: {(attn_values > 0).sum()} / {attn_values.size}")
        else:
            print("  ✗ No _attention_patterns_df attribute")
            
    except Exception as e:
        print(f"✗ Failed to extract attention patterns: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# 6. TEST NEIGHBOR ABLATION SCORES
# =============================================================================
print("\n" + "="*70)
print("6. TESTING NEIGHBOR ABLATION SCORES")
print("="*70)

if baseline_model is not None:
    try:
        cell_types = adata.obs['cell_type'].unique().tolist()
        print(f"Available cell types: {cell_types}")
        
        # Test for first receiver type
        receiver_ct = cell_types[1] if len(cell_types) > 1 else cell_types[0]
        sender_ct = cell_types[0]
        
        print(f"\nTesting ablation: {sender_ct} -> {receiver_ct}")
        
        ablation_scores = baseline_model.get_neighbor_ablation_scores(
            adata=adata,
            cell_type=receiver_ct,
            ablated_neighbor_ct_sub=[sender_ct],
            compute_z_value=True,
        )
        print("✓ Ablation scores computed")
        
        if hasattr(ablation_scores, '_ablation_scores_df'):
            abl_df = ablation_scores._ablation_scores_df
            print(f"  DataFrame shape: {abl_df.shape}")
            print(f"  Columns: {list(abl_df.columns)}")
            print(f"  Sample values:")
            print(abl_df.head())
        else:
            print("  ✗ No _ablation_scores_df attribute")
            
    except Exception as e:
        print(f"✗ Failed to compute ablation scores: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# 7. CHECK GROUND TRUTH
# =============================================================================
print("\n" + "="*70)
print("7. CHECKING GROUND TRUTH")
print("="*70)

gt_summary_path = os.path.join(DATA_DIR, "ground_truth_summary.json")
if os.path.exists(gt_summary_path):
    import json
    with open(gt_summary_path, 'r') as f:
        gt_summary = json.load(f)
    print(f"✓ Ground truth summary loaded")
    print(f"  Interactions: {list(gt_summary.get('interactions', {}).keys())}")
    print(f"  DE genes per interaction: {gt_summary.get('de_genes_per_interaction', {})}")
else:
    print(f"✗ Ground truth summary not found: {gt_summary_path}")

# Check for pickle ground truth
gt_pkl_path = os.path.join(DATA_DIR, f"replicate_{REPLICATE_IDX:02d}.ground_truth.pkl")
if os.path.exists(gt_pkl_path):
    import pickle
    with open(gt_pkl_path, 'rb') as f:
        gt = pickle.load(f)
    print(f"\n✓ Ground truth pickle loaded")
    print(f"  Type: {type(gt)}")
    if hasattr(gt, 'interactions'):
        print(f"  Interactions: {list(gt.interactions.keys())}")
    if hasattr(gt, 'de_genes'):
        print(f"  DE genes:")
        for k, v in gt.de_genes.items():
            if isinstance(v, pd.DataFrame):
                n_sig = (v['class'] == 1).sum() if 'class' in v.columns else 'unknown'
                print(f"    {k}: {len(v)} genes, {n_sig} significant")
            else:
                print(f"    {k}: {type(v)}")
    if hasattr(gt, 'interacting_cells'):
        print(f"  Interacting cells:")
        for k, v in gt.interacting_cells.items():
            print(f"    {k}: {len(v)} cells")
else:
    print(f"✗ Ground truth pickle not found: {gt_pkl_path}")
    print("  This is needed for proper evaluation!")

# =============================================================================
# 8. SUMMARY
# =============================================================================
print("\n" + "="*70)
print("8. SUMMARY & RECOMMENDATIONS")
print("="*70)

issues = []

if '_nn_idx' not in adata.obsm:
    issues.append("- Data missing _nn_idx: Call AMICI.setup_anndata() before evaluation")
    
if not os.path.exists(gt_pkl_path):
    issues.append("- Ground truth pickle missing: Regenerate data with pickle saving")
    
if baseline_model is None:
    issues.append("- Baseline model failed to load")
    
if ba_amici_model is None:
    issues.append("- BA-AMICI model failed to load")

if issues:
    print("\n⚠️  ISSUES FOUND:")
    for issue in issues:
        print(issue)
else:
    print("\n✓ All checks passed!")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)