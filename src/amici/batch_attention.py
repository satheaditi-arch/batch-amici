import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchAwareCrossAttention(nn.Module):
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

        # Standard projections
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)

        # Batch embeddings
        self.batch_emb = nn.Embedding(num_batches, hidden_dim)
        self.U_Q = nn.Linear(hidden_dim, hidden_dim)
        self.U_K = nn.Linear(hidden_dim, hidden_dim)

        # --- THE MISSING PART ---
        # Distance Coefficient Network
        self.dist_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_heads) 
        )
        # ------------------------

        # Optional batch-pair bias
        if use_batch_pair_bias:
            self.gamma = nn.Parameter(
                torch.zeros(num_batches, num_batches), requires_grad=True
            )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.normal_(self.batch_emb.weight, std=0.02)
        if self.use_batch_pair_bias:
            nn.init.zeros_(self.gamma)

    def forward(
        self,
        h_recv: torch.Tensor,       # [Batch, Dim]
        h_send: torch.Tensor,       # [Batch, Neighbors, Dim]
        batch_recv: torch.Tensor,   # [Batch]
        batch_send: torch.Tensor,   # [Batch, Neighbors]
        dist_matrix: torch.Tensor = None,  # [Batch, Neighbors]
        mask: torch.Tensor = None,
    ):
        # Get dimensions
        B, N_neighbors, _ = h_send.size()
        
        # --- A. Embed Batch Info ---
        E_recv = self.batch_emb(batch_recv)
        E_send = self.batch_emb(batch_send)

        # --- B. Compute Q, K, V with Batch Conditioning ---
        Q = self.W_Q(h_recv) + self.U_Q(E_recv)
        Q = Q.view(B, 1, self.num_heads, self.head_dim)

        K = self.W_K(h_send) + self.U_K(E_send)
        K = K.view(B, N_neighbors, self.num_heads, self.head_dim)
        
        V = self.W_V(h_send)
        V = V.view(B, N_neighbors, self.num_heads, self.head_dim)

        # --- C. Base Attention Scores (b0) ---
        scores = torch.einsum("bihd,bjhd->bijh", Q, K) / (self.head_dim ** 0.5)
        scores = scores.squeeze(1) # [Batch, Neighbors, Heads]

        # --- D. Learned Distance Decay (b1) with Softplus ---
        if dist_matrix is not None:
            receiver_state = h_recv + E_recv
            b1_raw = self.dist_net(receiver_state) # [Batch, Heads]
            b1 = F.softplus(b1_raw)
            
            b1 = b1.unsqueeze(1) # [Batch, 1, Heads]
            d = dist_matrix.unsqueeze(-1) # [Batch, Neighbors, 1]
            
            scores = scores - (b1 * d)

        # --- E. Batch-Pair Bias ---
        if self.use_batch_pair_bias:
            gamma_vals = self.gamma[batch_recv.unsqueeze(1), batch_send]
            scores = scores + gamma_vals.unsqueeze(-1)

        # --- F. Softmax & Aggregate ---
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))

        attn = torch.softmax(scores, dim=1) 
        
        z = torch.einsum("bnh,bnhd->bhd", attn, V)
        z = z.reshape(B, self.hidden_dim)

        return z, attn

    def batch_pair_regularizer(self):
        """
        L2 regularization term for γ, to add to your loss if use_batch_pair_bias=True.
        """
        if not self.use_batch_pair_bias:
            return torch.tensor(0.0, device=self.gamma.device)
        return (self.gamma ** 2).mean()
