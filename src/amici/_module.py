"""
BA-AMICI Module - FIXED VERSION

Key Fixes:
1. Baseline uses standard attention (no batch conditioning)
2. BA-AMICI uses BatchAwareCrossAttention (with batch conditioning)
3. Proper handling of neighbor batch indices
4. Clear separation between baseline and batch-aware modes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from einops import rearrange, reduce
from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from transformer_lens.hook_points import HookedRootModule, HookPoint

from .batch_attention import BatchAwareCrossAttention
from ._components import AttentionBlock, ResNetMLP
from ._constants import NN_REGISTRY_KEYS


class StandardCrossAttention(nn.Module):
    """
    Standard cross-attention WITHOUT batch conditioning.
    This is what the BASELINE should use.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        
        # Distance coefficient network (same as original AMICI)
        self.dist_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_heads)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
    
    def forward(
        self,
        h_recv: torch.Tensor,
        h_send: torch.Tensor,
        batch_recv: torch.Tensor = None,  # Ignored in baseline
        batch_send: torch.Tensor = None,  # Ignored in baseline
        dist_matrix: torch.Tensor = None,
        mask: torch.Tensor = None,
    ):
        """Standard attention - ignores batch information."""
        B, N_neighbors, _ = h_send.size()
        
        # Standard Q, K, V
        Q = self.W_Q(h_recv).view(B, 1, self.num_heads, self.head_dim)
        K = self.W_K(h_send).view(B, N_neighbors, self.num_heads, self.head_dim)
        V = self.W_V(h_send).view(B, N_neighbors, self.num_heads, self.head_dim)
        
        # Attention scores
        scores = torch.einsum("bihd,bjhd->bijh", Q, K) / (self.head_dim ** 0.5)
        scores = scores.squeeze(1)  # [B, N, H]
        
        # Distance decay
        if dist_matrix is not None:
            b1_raw = self.dist_net(h_recv)
            b1 = F.softplus(b1_raw).unsqueeze(1)
            d = dist_matrix.unsqueeze(-1)
            scores = scores - (b1 * d)
        
        # Mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        
        # Softmax and aggregate
        attn = torch.softmax(scores, dim=1)
        z = torch.einsum("bnh,bnhd->bhd", attn, V)
        z = z.reshape(B, self.hidden_dim)
        
        return z, attn
    
    def batch_pair_regularizer(self):
        """No batch-pair bias in standard attention."""
        return torch.tensor(0.0)


class AMICIModule(HookedRootModule, BaseModuleClass):
    """
    Fixed AMICI Module with clear baseline vs BA-AMICI distinction.
    
    Key Changes:
    1. use_batch_aware=False -> StandardCrossAttention (NO batch info)
    2. use_batch_aware=True -> BatchAwareCrossAttention (WITH batch info)
    3. Adversarial training only when use_adversarial=True
    """
    
    def __init__(
        self,
        n_genes: int,
        n_labels: int,
        empirical_ct_means: torch.Tensor,
        
        # Batch-awareness control
        n_batches: int = 1,
        use_batch_aware: bool = False,  # NEW: Controls attention type
        use_adversarial: bool = False,
        lambda_adv: float = 0.0,
        lambda_pair: float = 1e-3,
        
        # Architecture params
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
        
        # BA-AMICI flags
        self.n_batches = n_batches
        self.use_batch_aware = use_batch_aware  # NEW FLAG
        self.use_adversarial = use_adversarial
        self.lambda_adv = torch.tensor(lambda_adv)
        self.lambda_pair = lambda_pair
        self.current_epoch = 0
        self.max_epochs = 100
        
        self.register_buffer("ct_profiles", empirical_ct_means)
        
        # ==================== EMBEDDINGS ====================
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
        
        self.kv_embed = ResNetMLP(
            n_input=self.n_nn_embed,
            n_output=self.n_heads * self.n_query_dim,
            n_layers=2,
            n_hidden=self.n_kv_dim,
            dropout=0.0,
        )
        
        # ==================== ATTENTION LAYER ====================
        # THIS IS THE KEY FIX: Choose attention type based on use_batch_aware
        attention_hidden_dim = self.n_query_dim * self.n_heads
        
        if self.use_batch_aware:
            print("✓ Using BatchAwareCrossAttention (BA-AMICI mode)")
            self.attention_layer = BatchAwareCrossAttention(
                hidden_dim=attention_hidden_dim,
                num_heads=self.n_heads,
                num_batches=self.n_batches,
                use_batch_pair_bias=False,
            )
        else:
            print("✓ Using StandardCrossAttention (Baseline mode)")
            self.attention_layer = StandardCrossAttention(
                hidden_dim=attention_hidden_dim,
                num_heads=self.n_heads,
            )
        
        # ==================== ADVERSARIAL ====================
        if self.use_adversarial:
            from .adversarial import BatchDiscriminator
            self.discriminator = BatchDiscriminator(
                num_genes=n_genes,
                num_batches=n_batches,
                hidden_dim=128
            )
            print(f"✓ Adversarial discriminator initialized (lambda={lambda_adv})")
        
        # ==================== OUTPUT ====================
        self.linear_head = nn.Linear(
            self.n_heads * self.n_query_dim,
            self.n_genes,
            bias=False,
        )
        
        # Hooks
        self.hook_label_embed = HookPoint()
        self.hook_nn_embed = HookPoint()
        self.hook_final_residual = HookPoint()
        
        self.setup()
    
    def _get_inference_input(self, tensors):
        labels = tensors[REGISTRY_KEYS.LABELS_KEY]
        nn_X = tensors[NN_REGISTRY_KEYS.NN_X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        
        # --- FIX: Get neighbor batch indices ---
        # Note: We use "nn_batch" string directly, matching what we set in _model.py
        if NN_REGISTRY_KEYS.NN_BATCH_KEY in tensors:
            nn_batch = tensors[NN_REGISTRY_KEYS.NN_BATCH_KEY]
        else:
            nn_batch = None
        
        return {
            "labels": labels,
            "nn_X": nn_X,
            "batch_index": batch_index,
            "nn_batch": nn_batch,
        }
    
    def inference(self, labels, nn_X, batch_index, nn_batch=None):
        label_embed = self.hook_label_embed(
            rearrange(self.ct_embed(labels), "b 1 d -> b d")
        )
        nn_embed = self.hook_nn_embed(self.nn_embed(nn_X))
        
        return {
            "label_embed": label_embed,
            "nn_embed": nn_embed,
            "batch_index": batch_index,
            "nn_batch": nn_batch,
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
            "nn_batch": inference_outputs["nn_batch"],
        }
    
    @auto_move_data
    def generative(
        self,
        labels,
        label_embed,
        nn_embed,
        nn_dist,
        batch_index,
        nn_batch=None,
        return_attention_patterns=False,
    ):
        query_embed = self.query_embed(label_embed)
        kv_embed = self.kv_embed(nn_embed)
        
        # Receiver batch
        batch_recv = batch_index.squeeze(-1).long()
        
        # Sender batch - use actual neighbor batches if available
        if nn_batch is not None:
            batch_send = nn_batch.long()
        else:
            # Fallback: assume same batch as receiver (NOT IDEAL)
            batch_send = batch_recv.unsqueeze(1).expand(-1, nn_embed.shape[1])
        
        # Attention mask for dropout
        attention_mask = None
        if self.training and self.neighbor_dropout > 0.0:
            attention_mask = (
                torch.rand((kv_embed.shape[0], kv_embed.shape[1]), device=kv_embed.device)
                > self.neighbor_dropout
            ).int()
        
        # Apply attention
        residual_embed, attn_patterns = self.attention_layer(
            h_recv=query_embed,
            h_send=kv_embed,
            batch_recv=batch_recv,
            batch_send=batch_send,
            dist_matrix=nn_dist,
            mask=attention_mask,
        )
        
        # Output
        residual = self.hook_final_residual(self.linear_head(residual_embed).float())
        batch_ct_means = self.ct_profiles[labels.squeeze(-1)].squeeze()
        prediction = (batch_ct_means + residual).float()
        
        return {
            "residual_embed": residual_embed,
            "residual": residual,
            "prediction": prediction,
            "attention_patterns": attn_patterns,
        }
    
    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight=1.0):
        true_X = tensors[REGISTRY_KEYS.X_KEY]
        prediction = generative_outputs["prediction"]
        
        # Reconstruction loss
        reconstruction_loss = F.gaussian_nll_loss(
            prediction, true_X, var=torch.ones_like(prediction), reduction="none"
        ).sum(-1)
        
        # Attention entropy penalty
        attention_penalty = torch.zeros(true_X.shape[0], device=true_X.device)
        if self.attention_penalty_coef > 0.0:
            attention_patterns = generative_outputs["attention_patterns"]
            eps = torch.finfo(attention_patterns.dtype).eps
            attention_entropy_terms = (
                -1 * attention_patterns *
                torch.log(torch.clamp(attention_patterns, min=eps, max=1 - eps))
            )
            attention_penalty = reduce(
                reduce(attention_entropy_terms, "b h k -> b h", "sum"),
                "b h -> b",
                "mean",
            )
        
        # Adversarial loss
        adv_loss = torch.tensor(0.0, device=true_X.device)
        if self.use_adversarial and self.training:
            residuals = generative_outputs["residual"]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY].squeeze().long()
            
            alpha = self._get_grl_alpha()
            batch_logits = self.discriminator(residuals, alpha=alpha)
            adv_loss = F.cross_entropy(batch_logits, batch_index, reduction='mean')
        
        # Batch-pair regularization
        # gamma_reg = self.attention_layer.batch_pair_regularizer()
        
        # Total loss
        total_loss = torch.mean(
            reconstruction_loss +
            self.attention_penalty_coef * attention_penalty
        )
        
        if self.use_adversarial and self.training:
            total_loss = total_loss + self.lambda_adv * adv_loss
        
        if self.use_batch_aware:
            total_loss = total_loss + self.lambda_pair  # * gamma_reg
        
        return LossOutput(
            loss=total_loss,
            reconstruction_loss=reconstruction_loss,
            kl_local={
                "attention_penalty": self.attention_penalty_coef * attention_penalty,
                "adv_loss": adv_loss,
                # "gamma_reg": gamma_reg,
            },
            extra_metrics={
                "attention_penalty_coef": torch.tensor(self.attention_penalty_coef),
                "lambda_adv": self.lambda_adv,
                "grl_alpha": torch.tensor(self._get_grl_alpha() if self.use_adversarial else 0.0),
                "adv_loss_val": adv_loss.detach(),
                # "gamma_reg_val": gamma_reg.detach() if isinstance(gamma_reg, torch.Tensor) else torch.tensor(gamma_reg),
            },
        )
    
    def _get_grl_alpha(self):
        """Get GRL alpha based on training progress."""
        if self.max_epochs == 0:
            return 1.0
        progress = min(1.0, (2.0 * self.current_epoch) / self.max_epochs)
        return progress
    
    def on_train_epoch_end(self):
        """Called at end of each epoch to update scheduling."""
        self.current_epoch += 1