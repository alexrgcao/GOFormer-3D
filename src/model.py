import math
import torch
import torch.nn as nn

class ProteinFunctionModel(nn.Module):
    """
    Transformer-based model for protein function prediction that integrates:
      - ESM embeddings (seq_embed)
      - Additional single-residue structural features (pLDDT, centroid, orientation, side-chain)
      - Pairwise features (euclidean_distances, PAE, edge_vectors)
      - Optional relative position bias
    """
    def __init__(
        self, 
        d_emb=2560,       # ESM embedding dimension
        d_model=512, 
        n_heads=8, 
        dim_ff=1024, 
        num_layers=6, 
        num_go_terms=4273,
        max_len=5000,
        use_rel_pos_bias=True,
        use_struct_bias=True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_rel_pos_bias = use_rel_pos_bias
        
        self.input_proj = nn.Linear(d_emb, d_model)
        
        self.plddt_proj     = nn.Linear(1, d_model)
        self.centroid_proj  = nn.Linear(3, d_model)
        self.orient_proj    = nn.Linear(3, d_model)
        self.sidechain_proj = nn.Linear(3, d_model)

        self.fusion_proj = nn.Linear(5 * d_model, d_model)
        
        self.pairwise_proj = nn.Linear(5, 1)
        
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dim_ff, use_rel_pos_bias)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
        self.go_head = nn.Linear(d_model, num_go_terms)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.01)
        
        self.rel_pos_bias = RelativePositionBias(max_distance=128, num_buckets=32)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,
                seq_embed,                # [B, L, d_emb] ESM embeddings
                attn_mask=None,           # [B, L] boolean mask (True for padded)
                pae=None,                 # [B, L, L] predicted aligned error
                plddt=None,               # [B, L]
                centroid=None,            # [B, L, 3]
                orientation_vectors=None, # [B, L, 3]
                side_chain_vectors=None,  # [B, L, 3]
                euclidean_distances=None, # [B, L, L]
                edge_vectors=None         # [B, L, L, 3]
               ):
        """
        seq_embed:  [B, L, d_emb]
        pae:        [B, L, L]
        plddt:      [B, L]
        centroid:   [B, L, 3]
        orientation_vectors: [B, L, 3]
        side_chain_vectors:  [B, L, 3]
        euclidean_distances: [B, L, L]
        edge_vectors:        [B, L, L, 3]
        """
        B, L, _ = seq_embed.shape
        
        x_seq = self.input_proj(seq_embed)
        
        if plddt is not None:
            x_plddt = self.plddt_proj(plddt.unsqueeze(-1))  # [B, L, d_model]
        else:
            x_plddt = torch.zeros(B, L, self.d_model, device=x_seq.device, dtype=x_seq.dtype)
        
        if centroid is not None:
            x_cent = self.centroid_proj(centroid)  # [B, L, d_model]
        else:
            x_cent = torch.zeros(B, L, self.d_model, device=x_seq.device, dtype=x_seq.dtype)
        
        if orientation_vectors is not None:
            x_orient = self.orient_proj(orientation_vectors)  # [B, L, d_model]
        else:
            x_orient = torch.zeros(B, L, self.d_model, device=x_seq.device, dtype=x_seq.dtype)
        
        if side_chain_vectors is not None:
            x_sc = self.sidechain_proj(side_chain_vectors)    # [B, L, d_model]
        else:
            x_sc = torch.zeros(B, L, self.d_model, device=x_seq.device, dtype=x_seq.dtype)
        
        # Concatenate along feature dimension: [B, L, 5*d_model]
        fused = torch.cat([x_seq, x_plddt, x_cent, x_orient, x_sc], dim=-1)
        # Project back to d_model
        x = self.fusion_proj(fused)  # [B, L, d_model]
        
        if L <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :L, :]
        
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls_token, x], dim=1)  # [B, L+1, d_model]
        
        if attn_mask is not None:
            pad_mask = torch.cat([
                torch.zeros(B, 1, dtype=torch.bool, device=attn_mask.device),
                attn_mask
            ], dim=1)  # [B, L+1]
        else:
            pad_mask = None
        
        
        pairwise_bias = None
        if (euclidean_distances is not None) and (pae is not None) and (edge_vectors is not None):
            # pairwise_feat: [B, L, L, 5]
            # where the last dimension is [dist, pae, edge_x, edge_y, edge_z]
            pairwise_feat = torch.cat([
                euclidean_distances.unsqueeze(-1),    # [B, L, L, 1]
                pae.unsqueeze(-1),                   # [B, L, L, 1]
                edge_vectors                         # [B, L, L, 3]
            ], dim=-1)  # => [B, L, L, 5]
            
            pairwise_scalar = self.pairwise_proj(pairwise_feat)  # [B, L, L, 1]
            pairwise_scalar = pairwise_scalar.squeeze(-1)        # [B, L, L]
            
            zero_row = torch.zeros(B, 1, L, device=pairwise_scalar.device)
            pairwise_scalar = torch.cat([zero_row, pairwise_scalar], dim=1)  # pad row
            
            zero_col = torch.zeros(B, L+1, 1, device=pairwise_scalar.device)
            pairwise_scalar = torch.cat([zero_col, pairwise_scalar], dim=2)  # pad col
            
            pairwise_bias = pairwise_scalar  # [B, L+1, L+1]
        
        for layer in self.layers:
            x = layer(
                x,
                pad_mask=pad_mask,
                pairwise_bias=pairwise_bias,
                rel_pos_bias=self.rel_pos_bias if self.use_rel_pos_bias else None
            )
        
        x = self.norm(x)
        cls_repr = x[:, 0, :]
        
        go_logits = self.go_head(cls_repr)  # [B, num_go_terms]
        return go_logits


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, use_rel_pos_bias=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model)
        )
        
        self.use_rel_pos_bias = use_rel_pos_bias

    def forward(self, x, pad_mask=None, pairwise_bias=None, rel_pos_bias=None):
        """
        x:           [B, N, d_model]   (N = L+1, including CLS)
        pad_mask:    [B, N]   (True = padded => ignore)
        pairwise_bias: [B, N, N], optional, scalar attention bias from pairwise feats
        rel_pos_bias:  module or None; if present, we can add it too
        """
        B, N, _ = x.shape
        n_heads = self.self_attn.num_heads
        z = self.ln1(x)
        
        if (pairwise_bias is not None) or (rel_pos_bias is not None and self.use_rel_pos_bias):
            bias = torch.zeros(B, N, N, device=x.device, dtype=x.dtype)
            if pairwise_bias is not None:
                bias += pairwise_bias  # [B, N, N]
            if rel_pos_bias is not None and self.use_rel_pos_bias:
                rp = rel_pos_bias(N).squeeze(-1)
                bias += rp.unsqueeze(0)
            attn_mask = bias.unsqueeze(1).repeat(1, n_heads, 1, 1).reshape(B * n_heads, N, N)
        else:
            attn_mask = None
        
        if attn_mask is not None and pad_mask is not None:
            pad_mask = pad_mask.to(attn_mask.dtype)

        attn_out, _ = self.self_attn(z, z, z, attn_mask=attn_mask, key_padding_mask=pad_mask)
        x = x + self.dropout1(attn_out)
        
        z2 = self.ln2(x)
        ff_out = self.ff(z2)
        x = x + self.dropout2(ff_out)
        
        return x


class RelativePositionBias(nn.Module):
    """
    Example relative-position bias (bucket-based).
    If you do not want it, remove or set use_rel_pos_bias=False.
    """
    def __init__(self, max_distance=128, num_buckets=32):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

    def forward(self, L):
        device = self.relative_attention_bias.weight.device
        context_pos = torch.arange(L, dtype=torch.long, device=device)[:, None]
        memory_pos  = torch.arange(L, dtype=torch.long, device=device)[None, :]
        rel = memory_pos - context_pos                     # signed [L, L]

        half_buckets = self.num_buckets // 2
        sign_offset  = (rel < 0).long() * half_buckets      # 0 for i≤j, half_buckets for i>j
        dist         = rel.abs()                           # [L, L]

        # small distances
        is_small     = dist < half_buckets
        small_bucket = dist

        # large distances (log‑scaled)
        large        = dist.float().clamp(min=half_buckets, max=self.max_distance)
        log_range    = math.log(self.max_distance / half_buckets + 1e-8)
        large_bucket = (
            (torch.log(large / half_buckets + 1e-8) / log_range * (half_buckets - 1))
            .long()
            + half_buckets
        )

        # select per‑position bucket, then add direction offset
        bucket = torch.where(is_small, small_bucket, large_bucket)
        relative_bucket = (bucket + sign_offset).clamp(0, self.num_buckets - 1)  # [L, L]

        # lookup embeddings
        values = self.relative_attention_bias(relative_bucket)  # [L, L, 1]
        return values

