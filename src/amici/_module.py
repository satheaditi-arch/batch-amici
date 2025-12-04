import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
#from scvi import REGISTRY_KEYS
try:
    from scvi import REGISTRY_KEYS
except Exception:
    from scvi.data._constants import REGISTRY_KEYS

from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from transformer_lens.hook_points import HookedRootModule, HookPoint

from .batch_attention import BatchAwareCrossAttention
from .adversarial import BatchDiscriminator
from ._components import AttentionBlock, ResNetMLP
from ._constants import NN_REGISTRY_KEYS


class AMICIModule(HookedRootModule, BaseModuleClass):
    def __init__(
        self,
        n_genes: int,
        n_labels: int,
        empirical_ct_means: torch.Tensor,

        # --- BA-AMICI NEW ARGS ---
        n_batches: int = 1,
        use_adversarial: bool = False,
        lambda_adv: float = 0.0,
        # ------------------------

        n_label_embed: int = 32,
        n_kv_dim: int = 256,
        n_query_embed_hidden: int = 512,
        n_query_dim: int = 64,
        n_nn_embed: int = 256,
        n_nn_embed_hidden: int = 1024,
        n_pos_coef_mlp_hidden: int = 512,
        n_head_size: int = 16,
        n_heads: int = 4,
        neighbor_dropout: float = 0.1,
        attention_dummy_score: float = 3.0,
        attention_penalty_coef: float = 0.0,
        value_l1_penalty_coef: float = 0.0,
        pos_coef_offset: float = -2.0,
        distance_kernel_unit_scale: float = 1.0,
    ):
        super().__init__()

        self.n_genes = n_genes
        self.n_labels = n_labels
        self.n_label_embed = n_label_embed
        self.n_query_embed_hidden = n_query_embed_hidden
        self.n_query_dim = n_query_dim
        self.n_kv_dim = n_kv_dim
        self.n_nn_embed = n_nn_embed
        self.n_nn_embed_hidden = n_nn_embed_hidden
        self.n_pos_coef_mlp_hidden = n_pos_coef_mlp_hidden
        self.attention_dummy_score = attention_dummy_score
        self.neighbor_dropout = neighbor_dropout
        self.attention_penalty_coef = attention_penalty_coef
        self.value_l1_penalty_coef = value_l1_penalty_coef
        self.distance_kernel_unit_scale = distance_kernel_unit_scale
        self.pos_coef_offset = pos_coef_offset
        self.n_head_size = n_head_size
        self.n_heads = n_heads
        self.empirical_ct_means = empirical_ct_means

        # --- BA-AMICI FLAGS ---
        self.n_batches = n_batches
        self.use_adversarial = use_adversarial
        self.lambda_adv = torch.tensor(lambda_adv)

        self.register_buffer("ct_profiles", self.empirical_ct_means)

        # ------------------- Embeddings -------------------
        self.ct_embed = nn.Embedding(self.n_labels, self.n_label_embed)

        self.query_embed = ResNetMLP(
            n_input=self.n_label_embed,
            n_output=self.n_heads * self.n_query_dim,
            n_layers=2,
            n_hidden=self.n_query_embed_hidden,
            dropout=0.0,
        )

        self.nn_embed = ResNetMLP(
            n_input=self.n_genes,
            n_output=self.n_nn_embed,
            n_layers=2,
            n_hidden=self.n_nn_embed_hidden,
            dropout=0.0,
        )

        # ✅ CRITICAL SHAPE FIX: KV must match query dim
        self.kv_embed = ResNetMLP(
            n_input=self.n_nn_embed,
            n_output=self.n_heads * self.n_query_dim,
            n_layers=2,
            n_hidden=self.n_kv_dim,
            dropout=0.0,
        )

        # ------------------- Batch Aware Attention -------------------
        self.attention_layer = BatchAwareCrossAttention(
            hidden_dim=self.n_query_dim * self.n_heads,
            num_heads=self.n_heads,
            num_batches=self.n_batches,
            use_batch_pair_bias=True,
        )

        # ------------------- Adversarial Discriminator -------------------
        if self.use_adversarial:
            self.discriminator = BatchDiscriminator(
                num_genes=self.n_genes,
                num_batches=self.n_batches,
            )

        # ------------------- Output Head -------------------
        self.linear_head = nn.Linear(
            self.n_heads * self.n_query_dim,
            self.n_genes,
            bias=False,
        )

        # ------------------- Hooks -------------------
        self.hook_label_embed = HookPoint()
        self.hook_nn_embed = HookPoint()
        self.hook_final_residual = HookPoint()
        self.hook_pe_embed = HookPoint()

        self.setup()

    # ======================================================
    # ================= SCVI INTERFACE =====================
    # ======================================================

    def _get_inference_input(self, tensors):
        labels = tensors[REGISTRY_KEYS.LABELS_KEY]
        nn_X = tensors[NN_REGISTRY_KEYS.NN_X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        return {
            "labels": labels,
            "nn_X": nn_X,
            "batch_index": batch_index,
        }

    def inference(self, labels, nn_X, batch_index):
        label_embed = self.hook_label_embed(
            rearrange(self.ct_embed(labels), "b 1 d -> b d")
        )
        nn_embed = self.hook_nn_embed(self.nn_embed(nn_X))

        return {
            "label_embed": label_embed,
            "nn_embed": nn_embed,
            "batch_index": batch_index,
        }

    def _get_generative_input(self, tensors, inference_outputs):
        labels = tensors[REGISTRY_KEYS.LABELS_KEY]
        nn_dist = tensors[NN_REGISTRY_KEYS.NN_DIST_KEY]

        return {
            "labels": labels,
            "label_embed": inference_outputs["label_embed"],
            "nn_embed": inference_outputs["nn_embed"],
            "nn_dist": nn_dist,
            "batch_index": inference_outputs["batch_index"],
        }

    # ======================================================
    # ====================== GENERATIVE ====================
    # ======================================================

    @auto_move_data
    def generative(
        self,
        labels,
        label_embed,
        nn_embed,
        nn_dist,
        batch_index,
        return_attention_patterns=False,
        return_attention_scores=False,
        return_v=False,
    ):
        query_embed = self.query_embed(label_embed)
        kv_embed = self.kv_embed(nn_embed)

        batch_recv = batch_index.squeeze(-1).long()
        batch_send = batch_recv.unsqueeze(1).repeat(1, nn_embed.shape[1])

        attention_mask = None
        if self.training and self.neighbor_dropout > 0.0:
            attention_mask = (
                torch.rand((kv_embed.shape[0], kv_embed.shape[1]), device=kv_embed.device)
                > self.neighbor_dropout
            ).int()

        residual_embed, attn_patterns = self.attention_layer(
            h_recv=query_embed,
            h_send=kv_embed,
            batch_recv=batch_recv,
            batch_send=batch_send,
            dist_matrix=nn_dist,
            mask=attention_mask,
        )

        residual = self.hook_final_residual(self.linear_head(residual_embed).float())
        batch_ct_means = self.ct_profiles[labels.squeeze(-1)].squeeze()
        prediction = (batch_ct_means + residual).float()

        return {
            "residual_embed": residual_embed,
            "residual": residual,
            "prediction": prediction,
            "attention_patterns": attn_patterns,
            "attention_scores": attn_patterns,
            "attention_v": None,
            "pos_coefs": torch.zeros_like(attn_patterns),
        }

    # ======================================================
    # ======================== LOSS ========================
    # ======================================================

    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight=1.0):
        true_X = tensors[REGISTRY_KEYS.X_KEY]
        prediction = generative_outputs["prediction"]

        # 1. Reconstruction loss
        reconstruction_loss = F.gaussian_nll_loss(
            prediction, true_X, var=torch.ones_like(prediction), reduction="none"
        ).sum(-1)

        # 2. Attention entropy penalty
        attention_penalty = torch.zeros(true_X.shape[0], device=true_X.device)
        if self.attention_penalty_coef > 0.0:
            attention_patterns = generative_outputs["attention_patterns"]
            eps = torch.finfo(attention_patterns.dtype).eps
            attention_entropy_terms = (
                -1
                * attention_patterns
                * torch.log(torch.clamp(attention_patterns, min=eps, max=1 - eps))
            )
            attention_penalty = reduce(
                reduce(attention_entropy_terms, "b h k -> b h", "sum"),
                "b h -> b",
                "mean",
            )

        # 3. Value L1 penalty
        value_l1_penalty = torch.zeros(true_X.shape[0], device=true_X.device)

        # 4.  ADVERSARIAL LOSS
        adv_loss = torch.tensor(0.0, device=true_X.device)
        if self.use_adversarial:
            residuals = generative_outputs["residual"]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY].squeeze().long()
            batch_logits = self.discriminator(residuals, alpha=self.lambda_adv)
            adv_loss = F.cross_entropy(batch_logits, batch_index)

        # 4.5  γ Batch-pair bias regularization
        gamma_reg = torch.tensor(0.0, device=true_X.device)
        if hasattr(self.attention_layer, "batch_pair_regularizer"):
            gamma_reg = self.attention_layer.batch_pair_regularizer()

        # 5.  TOTAL LOSS
        loss = torch.mean(
            reconstruction_loss
            + self.attention_penalty_coef * attention_penalty
            + self.value_l1_penalty_coef * value_l1_penalty
            + adv_loss
            + 1e-3 * gamma_reg
        )

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kl_local={
                "attention_penalty": self.attention_penalty_coef * attention_penalty,
                "value_l1_penalty": self.value_l1_penalty_coef * value_l1_penalty,
                "adv_loss": adv_loss,
                "gamma_reg": gamma_reg,
            },
            extra_metrics={
                "attention_penalty_coef": torch.tensor(self.attention_penalty_coef),
                "adv_loss_val": adv_loss.detach(),
                "gamma_reg_val": gamma_reg.detach(),
            },
        )
