import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReversal(Function):
    """
    Gradient Reversal Layer (GRL).
    During forward propagation, it acts as an identity function.
    During backpropagation, it multiplies the gradient by -lambda_adv.
    This allows the encoder to 'unlearn' batch information while the 
    discriminator tries to learn it.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class BatchDiscriminator(nn.Module):
    """
    The Adversarial Network (Discriminator).
    It takes the predicted gene expression residuals (Delta) and tries to 
    predict which batch the cell came from.
    """
    def __init__(self, num_genes, num_batches, hidden_dim=128):
        super(BatchDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_genes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_batches) 
            # Output is logits for each batch class
        )

    def forward(self, residuals, alpha=1.0):
        # 1. Apply Gradient Reversal
        reversed_residuals = GradientReversal.apply(residuals, alpha)
        
        # 2. Predict Batch ID
        batch_logits = self.net(reversed_residuals)
        
        return batch_logits

import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReversal(Function):
    """
    Gradient Reversal Layer (GRL).
    During forward propagation, it acts as an identity function.
    During backpropagation, it multiplies the gradient by -lambda_adv.
    This allows the encoder to 'unlearn' batch information while the 
    discriminator tries to learn it.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class BatchDiscriminator(nn.Module):
    """
    The Adversarial Network (Discriminator).
    It takes the predicted gene expression residuals (Delta) and tries to 
    predict which batch the cell came from.
    """
    def __init__(self, num_genes, num_batches, hidden_dim=128):
        super(BatchDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_genes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_batches) 
            # Output is logits for each batch class
        )

    def forward(self, residuals, alpha=1.0):
        # 1. Apply Gradient Reversal
        reversed_residuals = GradientReversal.apply(residuals, alpha)
        
        # 2. Predict Batch ID
        batch_logits = self.net(reversed_residuals)
        
        return batch_logits
