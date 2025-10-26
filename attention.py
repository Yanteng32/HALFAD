import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size=128, heads=8, dropout=None):
        """
        Self-attention mechanism module initialization
        :param embed_size: Input feature dimension (default 128)
        :param heads: Number of heads in multi-head attention (default 8)
        """
        super(SelfAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.dropout = dropout

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # Use linear layers to implement Q, K, V projections
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)  # Final output linear layer
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, query, mask=None):
        """
        Forward pass
        :param values: Input value matrix, shape [batch_size, N, channel=128]
        :param keys: Input key matrix, shape [batch_size, N, channel=128]
        :param query: Input query matrix, shape [batch_size, N, channel=128]
        :param mask: Optional mask matrix to block attention weights
        :return: Self-attention output, shape [batch_size, N, channel=128]
        """
        N = query.shape[1]  # Sequence length
        batch_size = query.shape[0]

        # Split input into multiple heads
        values = values.view(batch_size, N, self.heads, self.head_dim)
        keys = keys.view(batch_size, N, self.heads, self.head_dim)
        queries = query.view(batch_size, N, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Compute dot product between Q and K, output [batch_size, heads, N, N]
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # Scale attention
        attention = energy / (self.embed_size ** (1/2))

        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))

        # Apply softmax normalization
        attention = torch.softmax(attention, dim=-1)

        # Compute weighted sum to get output
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            batch_size, N, self.embed_size
        )

        # Pass through final linear layer
        out = self.fc_out(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VisionTransformerBlock(nn.Module):
    def __init__(self, embed_size=128, heads=8, dropout=None):
        super(VisionTransformerBlock, self).__init__()
        self.self_attention = SelfAttention(embed_size, heads, dropout)
        self.layer_norm = nn.LayerNorm(embed_size)

        self.fc = nn.Linear(embed_size, embed_size)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
            self.layer_norm2 = nn.LayerNorm(embed_size)
        else:
            self.dropout = None

    def forward(self, x, mask=None):
        attention_out = self.self_attention(x, x, x, mask)
        attention_out = self.layer_norm(attention_out + x)  # Residual connection + layer norm
        forward_out = F.relu(self.fc(attention_out))
        if self.dropout is not None:
            forward_out = self.dropout(forward_out)
            return self.layer_norm2(forward_out + x)
        return self.layer_norm(forward_out + attention_out)

class MultimodalTransformerBlock(nn.Module):
    def __init__(self, embed_size=128, heads=8, dropout=None):
        super(MultimodalTransformerBlock, self).__init__()
        self.self_attention1 = SelfAttention(embed_size, heads, dropout)
        self.self_attention2 = SelfAttention(embed_size, heads, dropout)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.fc1 = nn.Linear(embed_size, embed_size)
        self.fc2 = nn.Linear(embed_size, embed_size)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
            self.layer_norm2 = nn.LayerNorm(embed_size)
        else:
            self.dropout = None

    def forward(self, x1, x2, mask=None):
        attention_out1 = self.self_attention2(x2, x2, x1, mask)
        attention_out2 = self.self_attention1(x1, x1, x2, mask)

        attention_out1 = self.layer_norm(attention_out1 + x1)  # Residual connection + layer norm
        attention_out2 = self.layer_norm(attention_out2 + x2)

        forward_out1 = F.gelu(self.fc1(attention_out1))
        forward_out2 = F.gelu(self.fc2(attention_out2))

        if self.dropout is not None:
            forward_out1 = self.dropout(forward_out1)
            forward_out2 = self.dropout(forward_out2)
            return self.layer_norm2(forward_out1 + x1), self.layer_norm2(forward_out2 + x2)

        return self.layer_norm(forward_out1 + attention_out1), self.layer_norm(forward_out2 + attention_out2)

if __name__ == '__main__':
    batch_size = 1
    image_features_flattened = torch.randn(batch_size, 40, 128)
    # Instantiate multimodal transformer block
    vit_block = MultimodalTransformerBlock(embed_size=128, heads=8)

    # Forward pass
    output1, output2 = vit_block(image_features_flattened, image_features_flattened)
    print(output1.shape)
    print(output2.shape)
    # Output shape: [batch_size, N, channels]