from .utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaLN(nn.Module):

    def __init__(self, latent_dim, embed_dim=None):
        super().__init__()
        if embed_dim is None:
            embed_dim = latent_dim
        self.emb_layers = nn.Sequential(
            # nn.Linear(embed_dim, latent_dim, bias=True),
            nn.SiLU(),
            zero_module(nn.Linear(embed_dim, 2 * latent_dim, bias=True)),
        )
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, T, D
        """
        # B, T, 2D
        emb_out = self.emb_layers(emb)
        # scale: B, T, D / shift: B, T, D
        scale, shift = torch.chunk(emb_out, 2, dim=-1)
        h = self.norm(h) * (1 + scale) + shift
        return h

# Relative Positional Encoding (learnable)
class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(RelativePositionalEncoding, self).__init__()
        # Learnable positional embeddings
        self.pe = nn.Parameter(torch.randn(max_len, d_model))
    
    def forward(self, x):
        # x shape: (B, T, d_model)
        T = x.size(1)
        return x + self.pe[:T, :].unsqueeze(0)
        
    
class RelativePositionalMultiHeadAttention(nn.Module):
    def __init__(self, latent_dim, num_head, dropout, max_len=4096):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_head = num_head
        self.dropout = dropout
        self.head_dim = latent_dim // num_head
        self.relative_positional_encoding = RelativePositionalEncoding(latent_dim, max_len)
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(latent_dim, 3 * latent_dim, bias=True)
        self.out = nn.Linear(latent_dim, latent_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        x: B, T, D
        """
        B, T, D = x.size()
        qkv = self.qkv(x) # B, T, 3D
        q, k, v = torch.chunk(qkv, 3, dim=-1) 
        q = q.view(B, T, self.num_head, self.head_dim).transpose(1, 2) # B, H, T, D
        k = k.view(B, T, self.num_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_head, self.head_dim).transpose(1, 2)
        pos_embedding = self.relative_positional_encoding(x) # B, T, D
        pos_embedding = pos_embedding.expand(B, -1, -1, -1) # B, T, T, D
        q = q + pos_embedding
        k = k + pos_embedding
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -1e9)
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.out(y)
        return y
    
class VanillaSelfAttention(nn.Module):
    def __init__(self, latent_dim, num_head, dropout, embed_dim=None, emcoding_mode='none'):
        super().__init__()
        self.num_head = num_head
        self.norm = AdaLN(latent_dim, embed_dim)
        self.encoding_mode = emcoding_mode
        if emcoding_mode == 'relative':
            self.attention = RelativePositionalMultiHeadAttention(latent_dim, num_head, dropout)
        elif emcoding_mode == 'none':
            self.attention = nn.MultiheadAttention(latent_dim, num_head, dropout=dropout, batch_first=True,
                                               add_zero_attn=True)
        else:
            raise ValueError('Unknown encoding mode')

    def forward(self, x, emb, key_padding_mask=None):
        """
        x: B, T, D
        emb: B, T, D
        """
        if emb is not None:
            x_norm = self.norm(x, emb)
        else:
            x_norm = x
        if self.encoding_mode == 'none':
            y = self.attention(x_norm, x_norm, x_norm,
                               attn_mask=None,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        else:
            y = self.attention(x_norm, attn_mask=None, 
                               key_padding_mask=key_padding_mask)
        return y


class VanillaCrossAttention(nn.Module):
    def __init__(self, latent_dim, xf_latent_dim, num_head, dropout, embed_dim=None, emcoding_mode='none'):
        super().__init__()
        self.num_head = num_head
        self.norm = AdaLN(latent_dim, embed_dim)
        self.xf_norm = AdaLN(xf_latent_dim, embed_dim)
        self.encoding_mode = emcoding_mode
        if emcoding_mode == 'relative':
            self.attention = RelativePositionalMultiHeadAttention(latent_dim, num_head, dropout)
        elif emcoding_mode == 'rope':
            self.attention = RoPEMultiHeadAttention(latent_dim, num_head, dropout)
        elif emcoding_mode == 'none':
            self.attention = nn.MultiheadAttention(latent_dim, num_head, dropout=dropout, batch_first=True,
                                               add_zero_attn=True)
        else:
            raise ValueError('Unknown encoding mode')

    def forward(self, x, xf, emb, key_padding_mask=None):
        """
        x: B, T, D
        xf: B, N, L
        """
        if emb is not None:
            x_norm = self.norm(x, emb)
            xf_norm = self.xf_norm(xf, emb)
        else:
            x_norm = x
            xf_norm = xf
        if self.encoding_mode == 'none':
            y = self.attention(x_norm, xf_norm, xf_norm,
                           attn_mask=None,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        else:
            y = self.attention(x_norm, attn_mask=None, 
                               key_padding_mask=key_padding_mask)
        return y


class FFN(nn.Module):
    def __init__(self, latent_dim, ffn_dim, dropout, embed_dim=None):
        super().__init__()
        self.norm = AdaLN(latent_dim, embed_dim)
        self.linear1 = nn.Linear(latent_dim, ffn_dim, bias=True)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim, bias=True))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, emb=None):
        if emb is not None:
            x_norm = self.norm(x, emb)
        else:
            x_norm = x
        y = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        return y


class FinalLayer(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.linear = zero_module(nn.Linear(latent_dim, out_dim, bias=True))

    def forward(self, x):
        x = self.linear(x)
        return x