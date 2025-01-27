import time
import os
import logging
import random
import argparse
from typing import Optional, Set


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image


logger = logging.getLogger(__name__)


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
        x = self.msa(x) + x
        x = self.mlp(x) + x
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
        x = self.pos_emb + x
        # TODO: dropout

        # feed patch into MLP
        # b, n, dim
        x = self.transformer(x)

        x = x[:, 0]
        x = self.classify_head(x)

        return x

def list_files(path):
    for root, _, files in os.walk(path):
        for f in files:
            yield os.path.join(root, f)
            break  # TODO remove

def dataset_folder_iter(
    dataset: str,
    offset: int,
    limit: Optional[int],
    shuffle: bool,
    filter_exts: Optional[Set[str]] = None,
    shuffle_fn=random.shuffle,
    **kwargs,
):
    print(
        f'dataset_folder_iter("{dataset}", offset={offset}, limit={limit}): retrieving files',
        flush=True
    )
    t1 = time.perf_counter_ns()
    # see: https://github.com/pytorch/vision/blob/d0ebeb55573820df89fa24f6418b9e624683932d/torchvision/datasets/folder.py#L36
    classes = sorted(entry.name for entry in os.scandir(dataset) if entry.is_dir())
    class_to_idx = {}
    if classes:
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    paths = list(list_files(dataset))
    paths.sort()
    t2 = time.perf_counter_ns()
    old_len = len(paths)
    if limit is not None:
        paths = paths[offset:offset+limit]
    else:
        paths = paths[offset:]
    assert len(paths) != 0, "no paths given"
    print(
        f'dataset_folder_iter("{dataset}", offset={offset}, limit={limit}) path expansion took {(t2-t1)/1e6:.3f}ms; num_paths={len(paths)}, total num_paths={old_len}',
        flush=True
    )
    # TODO: don't use this shuffle_fn
    if shuffle:
        shuffle_fn(paths)

    if filter_exts is None:
        filter_exts = set()

    for path in paths:
        bn, ext = os.path.splitext(path)
        dirname = os.path.dirname(bn)
        if filter_exts and ext not in filter_exts:
            continue

        dir = dirname.split(dataset)[1][1:]
        target = class_to_idx.get(dir)
        yield {
            ext: path,
            "uuid": os.path.basename(bn),
            "dir": dir,
            "target": target,
        }

class ImageNet(torch.utils.data.Dataset):
    def __init__(self, root, split, transform):
        super().__init__()
        self.root = root
        self.label_txt = os.path.join(root, "labels.txt")
        self.labels = [(x.split(","), i) for i, x in enumerate(open(self.label_txt).readlines())]
        self.labels = {x[0]: (i, x[1].strip()) for x, i in self.labels}
        self.data = list(dataset_folder_iter(
            os.path.join(self.root, split),
            0,
            None,
            shuffle=False,
        ))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x["image"] = Image.open(x[".JPEG"]).convert("RGB")
        x["image"] = self.transform(x["image"])
        x["target"] = self.labels[x["dir"]][0]
        return x


def train(args):
    train_dset = ImageNet(
        root="/datasets01/imagenet_full_size/061417/",
        split="train",
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    )
    train_dloader = torch.utils.data.DataLoader(
        train_dset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = ViT()
    model = model.train().to(args.device)

    optim = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    )
    torch.autograd.set_detect_anomaly(True)
    i = 0
    for x in train_dloader:
        optim.zero_grad()
        y = model(x["image"].to(args.device))
        target = x["target"].to(args.device)
        loss = F.cross_entropy(y, target)

        loss.backward()
        optim.step()
        if i % 10 == 0:
            print(f"[{i}] loss={loss:.2f}", flush=True)

        i += 1
        if i >= 1_000:
            break

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ViT',
        description='Trains a ViT',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    args = parser.parse_args()
    train(args)
