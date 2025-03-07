import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        
        # Create layers and layer norms
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(k) for k in h + [output_dim]
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, (layer, layer_norm) in enumerate(zip(self.layers, self.layer_norms)):
            x = layer(x)
            x = F.relu(x, inplace=True)
            x = layer_norm(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)  # Use `torch.sigmoid` instead of `F.sigmoid` as the latter is deprecated
        return x
    

class Attention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        # assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(self.embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, self.embedding_dim)

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, *rest_dims = x.shape
        c = rest_dims[0]
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)
    
    def _dim_expand(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return x

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                output_attn: bool = False) -> torch.Tensor:
        # assert k.shape == v.shape
        # Input projections
        q = self.q_proj(self._dim_expand(q))
        k = self.k_proj(self._dim_expand(k))
        v = self.v_proj(self._dim_expand(v))

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        b, n_heads, n_tokens, c_per_head = k.shape
        attn = torch.einsum("bhnc,bhmc->bhnm", q, k)   # B x N_heads x N_tokens x N_refer
        attn = attn / math.sqrt(c_per_head)

        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bhnm,bhmc->bhnc", attn, v) # B x N_heads x N_tokens x C_per_head
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        if output_attn:
            out_attn = attn
            return out, out_attn

        return out


class MultiWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        block_num: int = 0,
    ) -> None:
        super().__init__()

        self.block_num = block_num  # 0 for first layer, skip the first self attention
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_sparse_to_token = Attention(embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.cross_attn_dense_to_token = Attention(embedding_dim, num_heads)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(embedding_dim, embedding_dim, embedding_dim, 3)
        self.norm4 = nn.LayerNorm(embedding_dim)

    def with_pos(self, tokens, pos):
        return tokens if pos is None else tokens + pos

    def forward(self, tokens: Tensor, sparse_keys: Tensor, dense_keys: Tensor, dense_pos: Tensor = None):

        attn_out = self.self_attn(q=tokens, k=tokens, v=tokens)
        tokens = self.norm1(tokens + attn_out)

        # asign multi-modal information to tokens
        attn_out = self.cross_attn_sparse_to_token(q=tokens, k=sparse_keys, v=sparse_keys)  
        tokens = self.norm2(tokens + attn_out)

        # search object queries
        dense_keys_with_pos = self.with_pos(dense_keys, dense_pos)
        attn_out = self.cross_attn_dense_to_token(q=tokens, k=dense_keys_with_pos, v=dense_keys)    
        tokens = self.norm3(tokens + attn_out)

        # FFN block
        mlp_out = self.mlp(tokens)
        tokens = self.norm4(tokens + mlp_out)
        
        return tokens
    

class MaskAttentionModule(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        depth: int = 2,
        num_mask_tokens: int = 24
    ) -> None:
        super().__init__()

        self.num_mask_tokens = num_mask_tokens
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, embedding_dim)
        nn.init.xavier_uniform_(self.mask_tokens.weight)

        self.first_attn = Attention(embedding_dim, num_heads)
        self.first_norm = nn.LayerNorm(embedding_dim)

        self.attention_layers = nn.ModuleList()
        for i in range(depth):
            self.attention_layers.append(
                MultiWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    block_num=i,
                )
            )

        self.last_attn = Attention(embedding_dim=embedding_dim, num_heads=num_heads)
        self.last_norm = nn.LayerNorm(embedding_dim)

    def with_pos(self, tokens, pos):
        return tokens if pos is None else tokens + pos

    def forward(self, queries: Tensor, keys: Tensor, src: List = None, queries_pos: Tensor = None):
        """
        queries: proposal queries from point encoder [b, n, c]
        keys:    the target embedding from LLM       [b, 1, c]
        src:     scene embedding from point encoder: list[n, c]
        output:  mask_tokens contain both the mask information and the target information [b, m, c]
        """

        batch, _, dims = queries.shape
        mask_tokens = self.mask_tokens.weight.unsqueeze(0).expand(
            batch, -1, -1
            )

        queries_with_pos = self.with_pos(queries, queries_pos)
        attn_out = torch.cat([self.first_attn(q=q, k=k, v=k) for q, k in zip(queries_with_pos, src)], dim=0)  # pre align with scene information
        queries = self.first_norm(queries + attn_out)

        for layer in self.attention_layers:
            mask_tokens = layer(
                tokens=mask_tokens,
                sparse_keys=keys,
                dense_keys=queries,
                dense_pos=queries_pos,
            )

        # attn_out = self.last_attn(q=mask_tokens, k=keys, v=keys)
        # mask_tokens = self.last_norm(mask_tokens + attn_out)
        
        return mask_tokens