import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchAwareCrossAttention(nn.Module):
    """
    Batch-Aware Cross-Attention for BA-AMICI.
    
    Key Features:
    1. Batch embeddings condition Q, K, V projections
    2. Distance-dependent attention with learned decay coefficients
    3. Batch-pair bias matrix (γ) for systematic corrections
    4. Gradient-stable initialization and normalization
    
    Architecture:
        Q = W_Q(h_recv) + U_Q(E_batch_recv)
        K = W_K(h_send) + U_K(E_batch_send)
        V = W_V(h_send) + U_V(E_batch_send)  ← FIXED: Now batch-conditioned
        
        scores = (Q·K^T) / sqrt(d) - b1(h_recv, E_batch) * distance + γ[batch_recv, batch_send]
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_batches: int,
        use_batch_pair_bias: bool = True,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_batches = num_batches
        self.use_batch_pair_bias = use_batch_pair_bias

        # ==========================================
        # STANDARD PROJECTIONS
        # ==========================================
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)

        # ==========================================
        # BATCH CONDITIONING PROJECTIONS
        # ==========================================
        # Batch embeddings for each batch ID
        self.batch_emb = nn.Embedding(num_batches, hidden_dim)
        
        # Batch-conditioned transformations for Q, K, V
        self.U_Q = nn.Linear(hidden_dim, hidden_dim)
        self.U_K = nn.Linear(hidden_dim, hidden_dim)
        self.U_V = nn.Linear(hidden_dim, hidden_dim)  # ✅ ADDED: V conditioning

        # ==========================================
        # DISTANCE COEFFICIENT NETWORK
        # ==========================================
        # Learns how much distance matters for each head
        # b1(receiver_state) determines attention decay with distance
        self.dist_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_heads)
        )

        # ==========================================
        # BATCH-PAIR BIAS (γ Matrix)
        # ==========================================
        # Systematic correction for batch-pair interactions
        # γ[i,j] = bias when receiver is in batch i, sender in batch j
        if use_batch_pair_bias:
            self.gamma = nn.Parameter(
                torch.zeros(num_batches, num_batches), 
                requires_grad=True
            )

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters for stable training."""
        # Xavier initialization for projection matrices
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        
        # Batch conditioning matrices (smaller scale)
        nn.init.xavier_uniform_(self.U_Q.weight, gain=0.5)
        nn.init.xavier_uniform_(self.U_K.weight, gain=0.5)
        nn.init.xavier_uniform_(self.U_V.weight, gain=0.5)  # ✅ ADDED
        
        # Batch embeddings (small random initialization)
        nn.init.normal_(self.batch_emb.weight, std=0.02)
        
        # Gamma starts at zero (no correction initially)
        if self.use_batch_pair_bias:
            nn.init.zeros_(self.gamma)

    def forward(
        self,
        h_recv: torch.Tensor,       # [Batch, Dim] - Receiver cell features
        h_send: torch.Tensor,       # [Batch, Neighbors, Dim] - Sender cell features
        batch_recv: torch.Tensor,   # [Batch] - Batch ID of receivers
        batch_send: torch.Tensor,   # [Batch, Neighbors] - Batch ID of senders
        dist_matrix: torch.Tensor = None,  # [Batch, Neighbors] - Spatial distances
        mask: torch.Tensor = None,  # [Batch, Neighbors] - Attention mask
    ):
        """
        Forward pass of batch-aware cross-attention.
        
        Args:
            h_recv: Receiver cell embeddings [B, D]
            h_send: Sender cell embeddings [B, N, D]
            batch_recv: Batch indices for receivers [B]
            batch_send: Batch indices for senders [B, N]
            dist_matrix: Spatial distances [B, N] (optional)
            mask: Attention mask [B, N] (optional, 1=attend, 0=ignore)
        
        Returns:
            z: Aggregated features [B, D]
            attn: Attention weights [B, N, H]
        """
        B, N_neighbors, _ = h_send.size()
        
        # ==========================================
        # A. EMBED BATCH INFORMATION
        # ==========================================
        E_recv = self.batch_emb(batch_recv)  # [B, D]
        E_send = self.batch_emb(batch_send)  # [B, N, D]

        # ==========================================
        # B. COMPUTE Q, K, V WITH BATCH CONDITIONING
        # ==========================================
        # Queries: Conditioned on receiver's batch
        Q = self.W_Q(h_recv) + self.U_Q(E_recv)
        Q = Q.view(B, 1, self.num_heads, self.head_dim)  # [B, 1, H, D_h]

        # Keys: Conditioned on sender's batch
        K = self.W_K(h_send) + self.U_K(E_send)
        K = K.view(B, N_neighbors, self.num_heads, self.head_dim)  # [B, N, H, D_h]
        
        # Values: ✅ NOW CONDITIONED ON SENDER'S BATCH
        V = self.W_V(h_send) + self.U_V(E_send)
        V = V.view(B, N_neighbors, self.num_heads, self.head_dim)  # [B, N, H, D_h]

        # ==========================================
        # C. BASE ATTENTION SCORES (b0)
        # ==========================================
        # Standard scaled dot-product attention
        scores = torch.einsum("bihd,bjhd->bijh", Q, K) / (self.head_dim ** 0.5)
        scores = scores.squeeze(1)  # [B, N, H]

        # ==========================================
        # D. DISTANCE DECAY (b1) WITH SOFTPLUS
        # ==========================================
        if dist_matrix is not None:
            # Compute distance decay coefficient per head
            receiver_state = h_recv + E_recv  # Batch-conditioned receiver
            b1_raw = self.dist_net(receiver_state)  # [B, H]
            
            # Softplus ensures b1 > 0 (farther = lower attention)
            b1 = F.softplus(b1_raw)
            
            # Apply distance penalty: closer cells attend more
            b1 = b1.unsqueeze(1)  # [B, 1, H]
            d = dist_matrix.unsqueeze(-1)  # [B, N, 1]
            
            scores = scores - (b1 * d)  # [B, N, H]

        # ==========================================
        # E. BATCH-PAIR BIAS (γ)
        # ==========================================
        if self.use_batch_pair_bias:
            # γ[batch_recv, batch_send] adds systematic correction
            gamma_vals = self.gamma[batch_recv.unsqueeze(1), batch_send]  # [B, N]
            scores = scores + gamma_vals.unsqueeze(-1)  # [B, N, H]

        # ==========================================
        # F. SOFTMAX & AGGREGATE
        # ==========================================
        # Apply mask (set masked positions to -inf before softmax)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))

        # Attention weights (normalized per head)
        attn = torch.softmax(scores, dim=1)  # [B, N, H]
        
        # Weighted sum of values
        z = torch.einsum("bnh,bnhd->bhd", attn, V)  # [B, H, D_h]
        z = z.reshape(B, self.hidden_dim)  # [B, D]

        return z, attn

    def batch_pair_regularizer(self):
        """
        L2 regularization term for γ.
        
        Prevents γ from growing unbounded. Add to loss as:
            loss = ... + lambda_pair * self.batch_pair_regularizer()
        
        Returns:
            Scalar tensor: Mean squared value of γ
        """
        if not self.use_batch_pair_bias:
            return torch.tensor(0.0, device=self.gamma.device if hasattr(self, 'gamma') else 'cpu')
        return (self.gamma ** 2).mean()

    def get_attention_stats(self):
        """
        Get statistics about the learned attention parameters.
        Useful for debugging and interpretation.
        
        Returns:
            dict: Statistics including gamma norms, distance coefficients, etc.
        """
        stats = {
            'num_heads': self.num_heads,
            'num_batches': self.num_batches,
        }
        
        if self.use_batch_pair_bias:
            stats['gamma_l2_norm'] = torch.norm(self.gamma).item()
            stats['gamma_max'] = self.gamma.max().item()
            stats['gamma_min'] = self.gamma.min().item()
            stats['gamma_mean'] = self.gamma.mean().item()
        
        return stats
