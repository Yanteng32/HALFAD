from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn.functional as F
from torch.nn import AdaptiveAvgPool1d, AdaptiveMaxPool1d


def exists(val):
    """Check if value exists (not None)"""
    return val is not None


def default(val, d):
    """Return value if exists, otherwise return default"""
    return val if exists(val) else d


class PreNorm(nn.Module):
    """Pre-Layer Normalization module"""

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # Layer normalization
        self.fn = fn  # Function to apply after normalization

    def forward(self, x, **kwargs):
        """Apply layer norm then pass to function"""
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """FeedForward module with GELU activation"""

    def __init__(self, dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # Gaussian Error Linear Unit activation
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """Forward pass through feedforward network"""
        return self.net(x)


class Attention(nn.Module):
    """Multi-head Attention module"""

    def __init__(self, dim, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # Inner dimension for attention
        self.heads = heads
        self.scale = dim_head ** -0.5  # Scaling factor for attention scores

        self.attend = nn.Softmax(dim=-1)  # Softmax for attention weights
        self.to_q = nn.Linear(dim, inner_dim, bias=False)  # Query projection
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)  # Key/Value projection

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # Output projection
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, kv_include_self=False, similarity=None):
        """
        Forward pass for attention
        :param x: Input tensor [batch_size, seq_len, dim]
        :param context: Context tensor for cross-attention
        :param kv_include_self: Whether to include self in key/value
        :param similarity: Optional similarity matrix to bias attention
        :return: Attention output [batch_size, seq_len, dim]
        """
        b, n, _ = x.shape
        h = self.heads
        context = default(context, x)  # Default to self-attention if no context

        if kv_include_self:
            # Include self in key/value for cross-attention
            context = torch.cat((x, context), dim=1)

        # Project to query, key, value
        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        # Rearrange to [batch, heads, seq_len, dim_head]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # Compute attention scores
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # Add similarity bias if provided
        if similarity is not None:
            similarity = similarity.unsqueeze(1).unsqueeze(-1)  # Add head and last dim
            similarity = similarity.expand(b, h, n, n)  # Expand to match attention shape
            dots = dots + similarity  # Add similarity to attention scores

        attn = self.attend(dots)  # Apply softmax

        # Compute weighted values
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')  # Combine heads
        return self.to_out(out)  # Project to output dimension


class Transformer(nn.Module):
    """Transformer module with multiple layers"""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)  # Final layer normalization
        for _ in range(depth):
            # Each layer has attention and feedforward with pre-norm
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, context=None, similarity=None):
        """Forward pass through transformer layers"""
        for attn, ff in self.layers:
            # Attention with residual connection
            x = attn(x, context=context, similarity=similarity) + x
            # Feedforward with residual connection
            x = ff(x) + x
        return self.norm(x)  # Final normalization


class CrossTransformer_MOD_AVG(nn.Module):
    """Cross-modal Transformer with average pooling"""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # Paired transformers for two modalities
            self.layers.append(nn.ModuleList([
                Transformer(dim, 1, heads, dim_head, mlp_dim, dropout=dropout),
                Transformer(dim, 1, heads, dim_head, mlp_dim, dropout=dropout)
            ]))

        # Pooling operations for feature aggregation
        self.gap = nn.Sequential(
            Rearrange('b n d -> b d n'),  # Rearrange for pooling
            AdaptiveAvgPool1d(1),  # Global average pooling
            Rearrange('b d n -> b (d n)')  # Rearrange back
        )
        self.gmp = nn.Sequential(
            Rearrange('b n d -> b d n'),  # Rearrange for pooling
            AdaptiveMaxPool1d(1),  # Global max pooling
            Rearrange('b d n -> b (d n)')  # Rearrange back
        )

    def forward(self, tokens1, tokens2):
        """Forward pass for cross-modal transformer"""
        # Compute cosine similarity between modalities
        sim = F.cosine_similarity(tokens1, tokens2, dim=-1, eps=1e-08)

        # Process through paired transformer layers
        for tokens1_, tokens2_ in self.layers:
            tokens1 = tokens1_(tokens1, context=tokens2, similarity=sim) + tokens1
            tokens2 = tokens2_(tokens2, context=tokens1, similarity=sim) + tokens2

        # Apply pooling to both modalities
        tokens1_cls_avg = self.gap(tokens1)
        tokens1_cls_max = self.gmp(tokens1)
        tokens2_cls_avg = self.gap(tokens2)
        tokens2_cls_max = self.gmp(tokens2)

        # Concatenate all pooled features
        cls_token = torch.cat([tokens1_cls_avg, tokens1_cls_max, tokens2_cls_avg, tokens2_cls_max], dim=1)
        return cls_token, tokens1, tokens2

