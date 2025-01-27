import torch
import torch.nn as nn
import torch.nn.functional as F

class MSA(nn.Module):
    def __init__(self, dim, heads, dropout=0.0, fused = False):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim // heads
        self.fused = fused
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

        try:
            self.scale = self.inner_dim ** -0.5
        except:
            breakpoint()

        self.qkv = nn.Linear(dim, heads * self.inner_dim * 3, bias = False)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape

        x = self.norm(x)
        qkv = self.qkv(x)
        qkv = qkv.reshape(b, n, 3, self.heads, -1)

        # 0 1 2 3 4
        # b n 3 h d -> 3 b h n d
        qkv = qkv.permute(2, 0, 3, 1, 4)  #  TODO: why does timm permute in this way?
        q, k, v = qkv.unbind(dim=0)

        if self.fused:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout_p,
            )
        else:
            sa = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            sa = F.softmax(sa, dim=-1)
            x = torch.matmul(sa, v)
    
        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.msa = MSA(dim=dim, heads=heads, dropout=dropout, fused=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x += self.msa(x)
        x += self.mlp(x)
        return self.norm(x)

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, heads, dim, mlp_dim, dropout):
        super().__init__()
        self.encoder = nn.Sequential(*[
            TransformerBlock(dim=dim, heads=heads, mlp_dim=mlp_dim, dropout=dropout)
            for i in range(n_layers)
        ])

    def forward(self, x):
        return self.encoder(x)

class ViT(nn.Module):
    def __init__(self, p=16, mlp_dim=2048, dim=1024, heads=8, num_layers=8, dropout=0.0, num_classes=1000, w=224, h=224, c=3):
        super().__init__()

        self.p = p
        self.dim = dim
        self.c = c
        self.patch_dim = p * p * c
        self.N = (w*h) // (p*p)

        self.class_tok = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_emb = nn.Parameter(torch.randn(1, self.N + 1, dim))
        self.patch_emb = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.classify_head = nn.Linear(dim, num_classes)

        self.transformer = TransformerEncoder(
            n_layers=num_layers,
            heads=heads,
            dim=dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

    def forward(self, x):
        if len(x.shape) != 4:
            # c h w -> b c h w
            assert len(x.shape) == 3
            x = x.unsqueeze(0)
        b, c, h, w = x.shape

        # get patches
        x = x.view(b, -1, self.patch_dim)
        x = self.patch_emb(x)

        # add class tok and pos embeddings
        class_toks = self.class_tok.repeat(b, 1, 1)
        x = torch.cat((class_toks, x), dim=1)
        x += self.pos_emb
        # TODO: dropout

        # feed patch into MLP
        # b, n, dim
        x = self.transformer(x)

        x = x[:, 0]
        x = self.classify_head(x)

        return x


def train():
    pass

if __name__ == "__main__":
    msa = MSA(dim=768, heads=12)
    x = torch.randn(1, 16, 768)
    y = msa(x)
    
    vit = ViT()
    inp = torch.randn(3, 224, 224)
    y = vit(inp)
    print(y.shape)
