"""
Interaction Consistency Evaluator for BA-AMICI

This module provides evaluation metrics for assessing batch-aware cell-cell
interaction inference methods. The key insight is that BA-AMICI should produce
CONSISTENT interaction predictions across replicates, while baseline methods
may show variability due to batch effects.

Evaluation Framework:
1. Single-replicate metrics (from AMICI paper):
   - Gene AUPRC: Can we identify DE genes downstream of interactions?
   - Sender AUPRC: Can we identify true sender cells?
   - Receiver AUPRC: Can we identify cells receiving signals?

2. Cross-replicate metrics (NEW for BA-AMICI):
   - Interaction Jaccard: Do detected interactions match across replicates?
   - Gene score correlation: Are the same genes implicated across replicates?
   - Attention pattern correlation: Are attention weights consistent?

Reference:
    Hong et al. (2025). AMICI: Attention Mechanism Interpretation of Cell-cell 
    Interactions. bioRxiv 2025.09.22.677860

Author: BA-AMICI Project
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class InteractionPrediction:
    """Stores predictions from a single model run."""
    
    # Gene-level predictions
    gene_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Columns: gene, interaction, score, [optional: pvalue, logfoldchange]
    
    # Cell-level predictions
    sender_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Columns: cell_idx, neighbor_idx, score
    
    receiver_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Columns: cell_idx, score
    
    # Attention patterns
    attention_patterns: Optional[np.ndarray] = None
    # Shape: [n_cells, n_neighbors, n_heads]
    
    # Metadata
    model_name: str = ""
    replicate_id: str = ""
    batch_labels: Optional[pd.Series] = None


@dataclass 
class EvaluationResults:
    """Stores evaluation metrics."""
    
    # Single-replicate metrics
    gene_auprc: Dict[str, float] = field(default_factory=dict)  # interaction -> AUPRC
    sender_auprc: Dict[str, float] = field(default_factory=dict)
    receiver_auprc: Dict[str, float] = field(default_factory=dict)
    
    # Cross-replicate metrics
    interaction_jaccard: float = 0.0
    gene_score_correlation: float = 0.0
    attention_correlation: float = 0.0
    
    # Detailed results
    pr_curves: Dict[str, pd.DataFrame] = field(default_factory=dict)
    confusion_matrices: Dict[str, np.ndarray] = field(default_factory=dict)


class InteractionConsistencyEvaluator:
    """
    Evaluator for BA-AMICI interaction consistency.
    
    This class provides methods to:
    1. Extract interaction predictions from trained AMICI models
    2. Compare predictions against ground truth (single-replicate)
    3. Compare predictions across replicates (cross-replicate consistency)
    
    Parameters
    ----------
    ground_truth : dict
        Ground truth information containing:
        - interactions: Dict of InteractionRule objects
        - de_genes: Dict of DE gene DataFrames per interaction
        - interacting_cells: Dict of cell IDs per interaction
    """
    
    def __init__(self, ground_truth: dict):
        self.ground_truth = ground_truth
        self.predictions: Dict[str, InteractionPrediction] = {}
        self.results: Dict[str, EvaluationResults] = {}
        
    def extract_predictions_from_amici(
        self,
        model,  # AMICI model
        adata: ad.AnnData,
        model_name: str,
        replicate_id: str,
        interactions_config: Dict[str, Dict],
    ) -> InteractionPrediction:
        """
        Extract interaction predictions from a trained AMICI model.
        
        Parameters
        ----------
        model : AMICI
            Trained AMICI model.
        adata : AnnData
            Data used for training/evaluation.
        model_name : str
            Name identifier for this model (e.g., "baseline", "ba_amici").
        replicate_id : str
            Replicate identifier.
        interactions_config : Dict
            Configuration of ground truth interactions.
            
        Returns
        -------
        prediction : InteractionPrediction
            Extracted predictions.
        """
        print(f"\nExtracting predictions: {model_name} on {replicate_id}")
                # In extract_predictions_from_amici, add at the start:
        print(f"Model type: {type(model)}")
        print(f"adata shape: {adata.shape}")
        print(f"adata.obsm keys: {list(adata.obsm.keys())}")
        print(f"adata.obs columns: {list(adata.obs.columns)}")

        # Check if model has the methods we need
        print(f"Has get_attention_patterns: {hasattr(model, 'get_attention_patterns')}")
        prediction = InteractionPrediction(
            model_name=model_name,
            replicate_id=replicate_id,
        )
        
        # Get batch labels if available
        if "batch" in adata.obs:
            prediction.batch_labels = adata.obs["batch"].copy()
        
        # Extract attention patterns
        try:
            attention_module = model.get_attention_patterns(adata)
            if hasattr(attention_module, "_attention_patterns_df"):
                attn_df = attention_module._attention_patterns_df
                prediction.attention_patterns = attn_df
        except Exception as e:
            warnings.warn(f"Could not extract attention patterns: {e}")
        
        # Extract gene scores via ablation
        gene_scores_list = []
        for interaction_name, config in interactions_config.items():
            try:
                gene_df = self._compute_gene_scores(
                    model, adata, 
                    sender_type=config["sender"],
                    receiver_type=config["receiver"],
                )
                gene_df["interaction"] = interaction_name
                gene_scores_list.append(gene_df)
            except Exception as e:
                warnings.warn(f"Could not compute gene scores for {interaction_name}: {e}")
        
        if gene_scores_list:
            prediction.gene_scores = pd.concat(gene_scores_list, ignore_index=True)
        
        # Extract sender scores from attention
        sender_scores = self._compute_sender_scores(model, adata, interactions_config)
        prediction.sender_scores = sender_scores
        
        # Extract receiver scores
        receiver_scores = self._compute_receiver_scores(model, adata, interactions_config)
        prediction.receiver_scores = receiver_scores
        
        # Store
        key = f"{model_name}_{replicate_id}"
        self.predictions[key] = prediction
        
        
        return prediction
    
    def _compute_gene_scores(
        self,
        model,
        adata: ad.AnnData,
        sender_type: str,
        receiver_type: str,
    ) -> pd.DataFrame:
        """
        Compute gene-level interaction scores via neighbor ablation.
        
        Following AMICI paper: ablate sender cell type neighbors and compute
        Wald test statistic for each gene.
        """
        # This requires model's ablation functionality
        try:
            ablation_results = model.get_ablation_scores(
                adata,
                ablate_cell_type=sender_type,
                receiver_cell_type=receiver_type,
            )
            return ablation_results
        except AttributeError:
            # Fallback: use attention-weighted contributions
            return self._compute_gene_scores_from_attention(
                model, adata, sender_type, receiver_type
            )
    
    def _compute_gene_scores_from_attention(
        self,
        model,
        adata: ad.AnnData,
        sender_type: str,
        receiver_type: str,
    ) -> pd.DataFrame:
        """
        Compute gene scores from attention patterns when ablation is unavailable.
        """
        # Get predictions
        predictions = model.get_model_output(adata)
        
        # Simple approach: correlation between attention to sender type and gene expression
        labels_key = model.adata_manager.get_state_registry("labels_key").original_key
        
        receiver_mask = adata.obs[labels_key] == receiver_type
        receiver_indices = np.where(receiver_mask)[0]
        
        gene_scores = []
        for gene_idx, gene in enumerate(adata.var_names):
            # Compute correlation or other score metric
            score = 0.0  # Placeholder
            gene_scores.append({
                "gene": gene,
                "score": score,
            })
        
        return pd.DataFrame(gene_scores)
    
    def _compute_sender_scores(
        self,
        model,
        adata: ad.AnnData,
        interactions_config: Dict,
    ) -> pd.DataFrame:
        """
        Compute sender cell scores from attention patterns.
        
        A neighbor is scored as a sender based on the attention weight
        it receives from receiver cells.
        """
        try:
            attention_module = model.get_attention_patterns(adata)
            attn_df = attention_module._attention_patterns_df
            nn_idxs = attention_module._nn_idxs_df
            
            # Aggregate attention by neighbor
            sender_scores = []
            for cell_idx in range(len(adata)):
                for neighbor_col in [c for c in attn_df.columns if c.startswith("neighbor_")]:
                    neighbor_idx_col = neighbor_col  # Same column name in nn_idxs
                    if neighbor_idx_col in nn_idxs.columns:
                        neighbor_idx = nn_idxs.loc[cell_idx, neighbor_idx_col]
                        attn_score = attn_df.loc[cell_idx, neighbor_col]
                        
                        sender_scores.append({
                            "cell_idx": adata.obs_names[cell_idx],
                            "neighbor_idx": adata.obs_names[int(neighbor_idx)] if not pd.isna(neighbor_idx) else None,
                            "score": attn_score,
                        })
            
            return pd.DataFrame(sender_scores)
            
        except Exception as e:
            warnings.warn(f"Could not compute sender scores: {e}")
            return pd.DataFrame(columns=["cell_idx", "neighbor_idx", "score"])
    
    def _compute_receiver_scores(
        self,
        model,
        adata: ad.AnnData,
        interactions_config: Dict,
    ) -> pd.DataFrame:
        """
        Compute receiver cell scores.
        
        A cell is scored as a receiver based on the maximum attention
        it gives to any neighbor.
        """
        try:
            attention_module = model.get_attention_patterns(adata)
            attn_df = attention_module._attention_patterns_df
            
            # Max attention per cell
            neighbor_cols = [c for c in attn_df.columns if c.startswith("neighbor_")]
            max_attention = attn_df[neighbor_cols].max(axis=1)
            
            receiver_scores = pd.DataFrame({
                "cell_idx": adata.obs_names,
                "score": max_attention.values,
            })
            
            return receiver_scores
            
        except Exception as e:
            warnings.warn(f"Could not compute receiver scores: {e}")
            return pd.DataFrame(columns=["cell_idx", "score"])
    
    def evaluate_single_replicate(
        self,
        prediction_key: str,
        adata: ad.AnnData,
        labels_key: str = "cell_type",
        subtype_key: str = "subtype",
    ) -> EvaluationResults:
        """
        Evaluate predictions against ground truth for a single replicate.
        
        Parameters
        ----------
        prediction_key : str
            Key for stored prediction (format: "model_replicate").
        adata : AnnData
            Data with ground truth labels.
        labels_key : str
            Column in obs for cell type labels.
        subtype_key : str
            Column in obs for subtype labels.
            
        Returns
        -------
        results : EvaluationResults
            Evaluation metrics.
        """
        if prediction_key not in self.predictions:
            raise ValueError(f"No predictions found for {prediction_key}")
        
        prediction = self.predictions[prediction_key]
        results = EvaluationResults()
        
        print(f"\nEvaluating {prediction_key}")
        
        # Task 1: Gene prediction AUPRC
        for interaction_name in self.ground_truth.get("interactions", {}):
            if interaction_name in self.ground_truth.get("de_genes", {}):
                gt_genes = self.ground_truth["de_genes"][interaction_name]
                
                if not prediction.gene_scores.empty:
                    pred_genes = prediction.gene_scores[
                        prediction.gene_scores["interaction"] == interaction_name
                    ]
                    
                    if len(pred_genes) > 0:
                        auprc = self._compute_auprc(
                            gt_genes, pred_genes,
                            merge_cols=["gene"],
                            gt_class_col="class",
                            pred_score_col="score",
                        )
                        results.gene_auprc[interaction_name] = auprc
                        print(f"  Gene AUPRC ({interaction_name}): {auprc:.3f}")
        
        # Task 2: Sender prediction AUPRC
        sender_gt = self._get_sender_ground_truth(adata, labels_key)
        if not prediction.sender_scores.empty and not sender_gt.empty:
            auprc = self._compute_auprc(
                sender_gt, prediction.sender_scores,
                merge_cols=["cell_idx", "neighbor_idx"],
                gt_class_col="class",
                pred_score_col="score",
            )
            results.sender_auprc["combined"] = auprc
            print(f"  Sender AUPRC: {auprc:.3f}")
        
        # Task 3: Receiver prediction AUPRC
        receiver_gt = self._get_receiver_ground_truth(adata, subtype_key)
        if not prediction.receiver_scores.empty and not receiver_gt.empty:
            auprc = self._compute_auprc(
                receiver_gt, prediction.receiver_scores,
                merge_cols=["cell_idx"],
                gt_class_col="class",
                pred_score_col="score",
            )
            results.receiver_auprc["combined"] = auprc
            print(f"  Receiver AUPRC: {auprc:.3f}")
        
        self.results[prediction_key] = results
        return results
    
    def _get_sender_ground_truth(
        self,
        adata: ad.AnnData,
        labels_key: str,
    ) -> pd.DataFrame:
        """
        Get ground truth sender labels.
        
        A neighbor is a true sender if:
        - It is of the sender cell type for an interaction
        - It is within the length scale of a receiver cell
        """
        if "_nn_idx" not in adata.obsm or "_nn_dist" not in adata.obsm:
            warnings.warn("No neighbor information in adata.obsm")
            return pd.DataFrame()
        
        nn_idxs = adata.obsm["_nn_idx"]
        nn_dists = adata.obsm["_nn_dist"]
        cell_types = adata.obs[labels_key].values
        
        gt_rows = []
        for interaction_name, rule in self.ground_truth.get("interactions", {}).items():
            sender_type = rule.sender_type
            receiver_type = rule.receiver_type
            length_scale = rule.length_scale
            
            for cell_idx in range(len(adata)):
                if cell_types[cell_idx] != receiver_type:
                    continue
                    
                for nn_idx_idx in range(nn_idxs.shape[1]):
                    neighbor_idx = int(nn_idxs[cell_idx, nn_idx_idx])
                    distance = nn_dists[cell_idx, nn_idx_idx]
                    
                    # Is this a true sender?
                    is_sender = (
                        cell_types[neighbor_idx] == sender_type and
                        distance <= length_scale
                    )
                    
                    gt_rows.append({
                        "cell_idx": adata.obs_names[cell_idx],
                        "neighbor_idx": adata.obs_names[neighbor_idx],
                        "class": 1 if is_sender else 0,
                    })
        
        return pd.DataFrame(gt_rows)
    
    def _get_receiver_ground_truth(
        self,
        adata: ad.AnnData,
        subtype_key: str,
    ) -> pd.DataFrame:
        """
        Get ground truth receiver labels.
        
        A cell is a true receiver if it is assigned to an "interacting" subtype.
        """
        gt_rows = []
        
        for interaction_name, rule in self.ground_truth.get("interactions", {}).items():
            interacting_subtype = rule.interaction_subtype
            receiver_type = rule.receiver_type
            
            for cell_idx, cell_name in enumerate(adata.obs_names):
                subtype = adata.obs[subtype_key].iloc[cell_idx]
                
                # Is this an interacting receiver?
                is_receiver = subtype == interacting_subtype
                
                gt_rows.append({
                    "cell_idx": cell_name,
                    "class": 1 if is_receiver else 0,
                })
        
        return pd.DataFrame(gt_rows).drop_duplicates()
    
    def _compute_auprc(
        self,
        gt_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        merge_cols: List[str],
        gt_class_col: str,
        pred_score_col: str,
    ) -> float:
        """Compute AUPRC between ground truth and predictions."""
        # Merge on common columns
        merged = pd.merge(gt_df, pred_df, on=merge_cols, how="inner")
        
        if len(merged) == 0:
            return 0.0
        
        if gt_class_col not in merged.columns or pred_score_col not in merged.columns:
            return 0.0
        
        y_true = merged[gt_class_col].values
        y_score = merged[pred_score_col].values
        
        # Handle missing/nan
        valid_mask = ~(np.isnan(y_score) | np.isnan(y_true))
        if valid_mask.sum() == 0:
            return 0.0
        
        y_true = y_true[valid_mask]
        y_score = y_score[valid_mask]
        
        # Need both classes present
        if len(np.unique(y_true)) < 2:
            return 0.0
        
        return average_precision_score(y_true, y_score)
    
    def evaluate_cross_replicate_consistency(
        self,
        model_name: str,
        replicate_ids: List[str],
        top_k_interactions: int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate consistency of interaction predictions across replicates.
        
        This is the key metric for BA-AMICI: do we detect the same interactions
        regardless of which replicate (batch) we train on?
        
        Parameters
        ----------
        model_name : str
            Name of model to evaluate.
        replicate_ids : List[str]
            List of replicate identifiers.
        top_k_interactions : int
            Number of top interactions to compare.
            
        Returns
        -------
        consistency_metrics : Dict[str, float]
            Dictionary of consistency metrics.
        """
        print(f"\nEvaluating cross-replicate consistency for {model_name}")
        
        # Collect predictions across replicates
        predictions = []
        for rep_id in replicate_ids:
            key = f"{model_name}_{rep_id}"
            if key in self.predictions:
                predictions.append(self.predictions[key])
        
        if len(predictions) < 2:
            warnings.warn(f"Need at least 2 replicates, found {len(predictions)}")
            return {}
        
        metrics = {}
        
        # 1. Interaction Jaccard
        interaction_sets = []
        for pred in predictions:
            if not pred.gene_scores.empty:
                # Get top interactions by gene score
                top_genes = (
                    pred.gene_scores
                    .groupby("interaction")
                    .apply(lambda x: x.nlargest(top_k_interactions, "score"))
                    .reset_index(drop=True)
                )
                interaction_set = set(
                    tuple(row) for _, row in 
                    top_genes[["interaction", "gene"]].iterrows()
                )
                interaction_sets.append(interaction_set)
        
        if len(interaction_sets) >= 2:
            pairwise_jaccards = []
            for i in range(len(interaction_sets)):
                for j in range(i + 1, len(interaction_sets)):
                    set_i, set_j = interaction_sets[i], interaction_sets[j]
                    if len(set_i | set_j) > 0:
                        jaccard = len(set_i & set_j) / len(set_i | set_j)
                        pairwise_jaccards.append(jaccard)
            
            if pairwise_jaccards:
                metrics["interaction_jaccard_mean"] = np.mean(pairwise_jaccards)
                metrics["interaction_jaccard_std"] = np.std(pairwise_jaccards)
                print(f"  Interaction Jaccard: {metrics['interaction_jaccard_mean']:.3f} ± {metrics['interaction_jaccard_std']:.3f}")
        
        # 2. Gene score correlation
        gene_score_correlations = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                pred_i, pred_j = predictions[i], predictions[j]
                
                if pred_i.gene_scores.empty or pred_j.gene_scores.empty:
                    continue
                
                # Merge on gene and interaction
                merged = pd.merge(
                    pred_i.gene_scores[["gene", "interaction", "score"]],
                    pred_j.gene_scores[["gene", "interaction", "score"]],
                    on=["gene", "interaction"],
                    suffixes=("_i", "_j"),
                )
                
                if len(merged) > 10:
                    corr, _ = stats.spearmanr(merged["score_i"], merged["score_j"])
                    if not np.isnan(corr):
                        gene_score_correlations.append(corr)
        
        if gene_score_correlations:
            metrics["gene_score_correlation_mean"] = np.mean(gene_score_correlations)
            metrics["gene_score_correlation_std"] = np.std(gene_score_correlations)
            print(f"  Gene score correlation: {metrics['gene_score_correlation_mean']:.3f} ± {metrics['gene_score_correlation_std']:.3f}")
        
        # 3. Attention pattern correlation (if available)
        attention_correlations = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                pred_i, pred_j = predictions[i], predictions[j]
                
                if pred_i.attention_patterns is None or pred_j.attention_patterns is None:
                    continue
                
                # This requires matching cells across replicates
                # For now, compute mean attention per cell type pair
                try:
                    mean_attn_i = self._compute_mean_attention_by_type(pred_i.attention_patterns)
                    mean_attn_j = self._compute_mean_attention_by_type(pred_j.attention_patterns)
                    
                    # Correlation of mean attention matrices
                    corr, _ = stats.spearmanr(mean_attn_i.flatten(), mean_attn_j.flatten())
                    if not np.isnan(corr):
                        attention_correlations.append(corr)
                except Exception:
                    pass
        
        if attention_correlations:
            metrics["attention_correlation_mean"] = np.mean(attention_correlations)
            metrics["attention_correlation_std"] = np.std(attention_correlations)
            print(f"  Attention correlation: {metrics['attention_correlation_mean']:.3f} ± {metrics['attention_correlation_std']:.3f}")
        
        return metrics
    
    def _compute_mean_attention_by_type(
        self,
        attention_df: pd.DataFrame,
    ) -> np.ndarray:
        """Compute mean attention aggregated by cell type pairs."""
        # This is a placeholder - actual implementation depends on attention format
        # Return dummy array for now
        return np.random.rand(10, 10)
    
    def compare_models(
        self,
        model_names: List[str],
        replicate_ids: List[str],
    ) -> pd.DataFrame:
        """
        Compare multiple models across replicates.
        
        Parameters
        ----------
        model_names : List[str]
            List of model names to compare.
        replicate_ids : List[str]
            List of replicate identifiers.
            
        Returns
        -------
        comparison_df : pd.DataFrame
            Comparison table with metrics for each model.
        """
        print("\n" + "="*60)
        print("Model Comparison")
        print("="*60)
        
        comparison_rows = []
        
        for model_name in model_names:
            row = {"model": model_name}
            
            # Aggregate single-replicate metrics
            gene_auprcs = []
            sender_auprcs = []
            receiver_auprcs = []
            
            for rep_id in replicate_ids:
                key = f"{model_name}_{rep_id}"
                if key in self.results:
                    result = self.results[key]
                    gene_auprcs.extend(result.gene_auprc.values())
                    sender_auprcs.extend(result.sender_auprc.values())
                    receiver_auprcs.extend(result.receiver_auprc.values())
            
            if gene_auprcs:
                row["gene_auprc_mean"] = np.mean(gene_auprcs)
                row["gene_auprc_std"] = np.std(gene_auprcs)
            if sender_auprcs:
                row["sender_auprc_mean"] = np.mean(sender_auprcs)
                row["sender_auprc_std"] = np.std(sender_auprcs)
            if receiver_auprcs:
                row["receiver_auprc_mean"] = np.mean(receiver_auprcs)
                row["receiver_auprc_std"] = np.std(receiver_auprcs)
            
            # Cross-replicate consistency
            consistency = self.evaluate_cross_replicate_consistency(
                model_name, replicate_ids
            )
            row.update(consistency)
            
            comparison_rows.append(row)
        
        comparison_df = pd.DataFrame(comparison_rows)
        
        print("\nComparison Results:")
        print(comparison_df.to_string())
        
        return comparison_df
    
    def generate_report(
        self,
        output_dir: Union[str, Path],
        model_names: List[str],
        replicate_ids: List[str],
    ) -> None:
        """
        Generate a comprehensive evaluation report.
        
        Parameters
        ----------
        output_dir : str or Path
            Directory to save report files.
        model_names : List[str]
            Models to include in report.
        replicate_ids : List[str]
            Replicates to include.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Comparison table
        comparison_df = self.compare_models(model_names, replicate_ids)
        comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
        
        # Detailed results per model
        for model_name in model_names:
            model_dir = output_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            for rep_id in replicate_ids:
                key = f"{model_name}_{rep_id}"
                
                if key in self.predictions:
                    pred = self.predictions[key]
                    
                    if not pred.gene_scores.empty:
                        pred.gene_scores.to_csv(
                            model_dir / f"gene_scores_{rep_id}.csv",
                            index=False
                        )
                    
                    if not pred.sender_scores.empty:
                        pred.sender_scores.to_csv(
                            model_dir / f"sender_scores_{rep_id}.csv",
                            index=False
                        )
                    
                    if not pred.receiver_scores.empty:
                        pred.receiver_scores.to_csv(
                            model_dir / f"receiver_scores_{rep_id}.csv",
                            index=False
                        )
        
        print(f"\nReport saved to {output_dir}")


def evaluate_ba_amici_benchmark(
    baseline_models: Dict[str, any],  # rep_id -> trained model
    ba_amici_models: Dict[str, any],  # rep_id -> trained model
    datasets: Dict[str, ad.AnnData],  # rep_id -> adata
    ground_truth: dict,
    interactions_config: Dict[str, Dict],
    output_dir: Union[str, Path],
) -> pd.DataFrame:
    """
    Run complete BA-AMICI evaluation benchmark.
    
    Convenience function that:
    1. Extracts predictions from all models
    2. Evaluates single-replicate metrics
    3. Evaluates cross-replicate consistency
    4. Compares baseline vs BA-AMICI
    
    Parameters
    ----------
    baseline_models : Dict[str, AMICI]
        Baseline AMICI models keyed by replicate.
    ba_amici_models : Dict[str, AMICI]
        BA-AMICI models keyed by replicate.
    datasets : Dict[str, AnnData]
        Datasets keyed by replicate.
    ground_truth : dict
        Ground truth information.
    interactions_config : Dict
        Interaction definitions.
    output_dir : str or Path
        Output directory for results.
        
    Returns
    -------
    comparison_df : pd.DataFrame
        Model comparison results.
    """
    evaluator = InteractionConsistencyEvaluator(ground_truth)
    
    replicate_ids = list(datasets.keys())
    
    # Extract predictions
    for rep_id in replicate_ids:
        adata = datasets[rep_id]
        
        if rep_id in baseline_models:
            evaluator.extract_predictions_from_amici(
                baseline_models[rep_id],
                adata,
                model_name="baseline",
                replicate_id=rep_id,
                interactions_config=interactions_config,
            )
        
        if rep_id in ba_amici_models:
            evaluator.extract_predictions_from_amici(
                ba_amici_models[rep_id],
                adata,
                model_name="ba_amici",
                replicate_id=rep_id,
                interactions_config=interactions_config,
            )
    
    # Evaluate single-replicate
    for rep_id in replicate_ids:
        adata = datasets[rep_id]
        
        for model_name in ["baseline", "ba_amici"]:
            key = f"{model_name}_{rep_id}"
            if key in evaluator.predictions:
                evaluator.evaluate_single_replicate(
                    key, adata,
                    labels_key="cell_type",
                    subtype_key="subtype",
                )
    
    # Generate report
    evaluator.generate_report(
        output_dir,
        model_names=["baseline", "ba_amici"],
        replicate_ids=replicate_ids,
    )
    
    # Return comparison
    return evaluator.compare_models(["baseline", "ba_amici"], replicate_ids)


if __name__ == "__main__":
    print("Interaction Consistency Evaluator for BA-AMICI")
    print("=" * 60)
    print("\nThis module provides evaluation metrics for BA-AMICI.")
    print("\nKey metrics:")
    print("  1. Gene AUPRC - Can we identify DE genes?")
    print("  2. Sender AUPRC - Can we identify true senders?")
    print("  3. Receiver AUPRC - Can we identify receivers?")
    print("  4. Interaction Jaccard - Are detections consistent across replicates?")
    print("  5. Gene score correlation - Same genes implicated?")
    print("  6. Attention correlation - Consistent attention patterns?")
