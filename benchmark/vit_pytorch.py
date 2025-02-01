import sys
import traceback
import json
import time
import os
import logging
import random
import argparse
import functools
from typing import Optional, Set
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter
from vit_pt_lucidrains import ViT as ViTRef


DSET_CACHE_DIR = "./datasets/"

logger = logging.getLogger(__name__)

def count_parameters(model):
    total_params = 0
    for p in model.parameters():
        if p.requires_grad:
            total_params += p.numel()
    return total_params

# taken from: https://github.com/mlfoundations/open_clip/blob/main/src/training/zero_shot.py#L29
def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]

class RollingAvg:
    def __init__(self, n):
        self.n = n
        self.data = []

    def add(self, x):
        self.data.append(x)
        if len(self.data) >= self.n:
            self.data = self.data[1:]

    def get(self):
        return sum(self.data) / len(self.data)

class MSA(nn.Module):
    def __init__(self, dim, heads, dropout=0.0, fused = False):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim // heads
        self.fused = fused
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

        self.scale = self.inner_dim ** -0.5

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
                scale=self.scale,
                dropout_p=self.dropout_p,
            )
        else:
            sa = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            sa = F.softmax(sa, dim=-1)
            sa = self.dropout(sa)
            x = torch.matmul(sa, v)
    
        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.0):
        super().__init__()
        # self.msa = MSA(dim=dim, heads=heads, dropout=dropout, fused=False)
        # TODO: arg for fused
        self.msa = MSA(dim=dim, heads=heads, dropout=dropout, fused=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.msa(x) + x
        x = self.mlp(x) + x
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, heads, dim, mlp_dim, dropout):
        super().__init__()
        self.encoder = nn.Sequential(*[
            TransformerBlock(dim=dim, heads=heads, mlp_dim=mlp_dim, dropout=dropout)
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.encoder(x)
        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, patch_size=16, mlp_dim=2048, dim=1024, heads=8, depth=8, dropout=0.0, num_classes=1000, image_size=224, c=3):
        super().__init__()

        self.patch_size = patch_size
        self.dim = dim
        self.c = c
        self.patch_dim = patch_size * patch_size * c
        self.N = (image_size*image_size) // (patch_size*patch_size)

        self.class_tok = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_emb = nn.Parameter(torch.randn(1, self.N + 1, dim))
        self.patch_emb = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.classify_head = nn.Linear(dim, num_classes)
        self.dropout = nn.Dropout(dropout)

        self.transformer = TransformerEncoder(
            n_layers=depth,
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
        x = x.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)
        # 0  1   2   3   4   5
        # b, c, nh, nw, ps, ps -> b, nh, nw, c, ps, ps
        x = x.permute(0, 2, 3, 4, 5, 1).reshape(b, -1, self.patch_size, self.patch_size, c)
        n = x.shape[1]
        x = x.reshape(b, n, self.patch_dim)
        x = self.patch_emb(x)

        # add class tok and pos embeddings
        class_toks = self.class_tok.repeat(b, 1, 1)
        x = torch.cat((class_toks, x), dim=1)
        # x += self.pos_emb[:, :(n+1)]
        x += self.pos_emb
        x = self.dropout(x)

        # feed patch into MLP
        x = self.transformer(x)

        x = x[:, 0]
        x = self.classify_head(x)

        return x

def list_files(path):
    for root, _, files in os.walk(path):
        for f in files:
            yield os.path.join(root, f)

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


def local_map(xs, map_fn, num_workers, process=True, progress=False):
    executor_class = ProcessPoolExecutor
    if not process:
        executor_class = ThreadPoolExecutor
    with executor_class(num_workers) as pool:
        for x in tqdm(pool.map(map_fn, xs), total=len(xs)):
            yield x

def _imagenet_load_and_get_target(x):
    if not os.path.exists(x[".JPEG"]):
        return None
    try:
        _ = Image.open(x[".JPEG"]).convert("RGB")
    except:
        print(traceback.format_exc())
        return None
    return x

def create_imagenet_dset_cache(args, split):
    root = args.imagenet_root
    label_txt_path = os.path.join(root, "labels.txt")
    labels = [(x.split(","), i) for i, x in enumerate(open(label_txt_path).readlines())]
    labels = {x[0]: (i, x[1].strip()) for x, i in labels}

    cache_path = os.path.join(DSET_CACHE_DIR, f"imagenet-{split}.jsonl")
    os.makedirs(DSET_CACHE_DIR, exist_ok=True)

    data = list(dataset_folder_iter(
        os.path.join(root, split),
        0,
        None,
        shuffle=False,
    ))

    map_fn = _imagenet_load_and_get_target
    with open(cache_path, "w") as out_f:
        print("Writing cache...")
        count = 0
        for x in local_map(data, map_fn, num_workers=args.num_workers, progress=True):
            if x is None:
                continue

            x["target"] = labels[x["dir"]][0]
            out_f.write(json.dumps(x) + "\n")
            count += 1
        print(f"{len(data)} -> {count}", flush=True)


def create_dset_cache(args):
    #create_imagenet_dset_cache(args, "train")
    create_imagenet_dset_cache(args, "val")


# TODO: generalize
class JsonlDset(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        super().__init__()
        self.data = [json.loads(x) for x in open(path).readlines()]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        try:
            x["image"] = Image.open(x[".JPEG"]).convert("RGB")
        except:
            x["image"] = torch.zeros(3, 256, 256)

        x["image"] = self.transform(x["image"])
        return x

def get_dataset(args, split, transform):
    return JsonlDset(
        path=os.path.join(DSET_CACHE_DIR, f"imagenet-{split}.jsonl"),
        transform=transform,
    )


def check_dataset(args):
    dset = get_dataset(args, "train", None)
    for i in tqdm(range(len(dset)), total=len(dset)):
        try:
            x = dset[i]
        except:
            print(i, traceback.format_exc())
            continue


def train(args):
    if args.checkpoint_dir is None:
        print("--checkpoint_dir not provided")
        sys.exit(1)

    if args.ref:
        model = ViTRef(
            image_size = args.img_size,
            patch_size = 32,
            num_classes = 1000,
            depth = 12,
            heads = 12,
            dim = 768,
            mlp_dim = 3072,
            dropout = 0.1,
        )
    else:
        model = ViT(
            image_size = args.img_size,
            num_classes = 1000,
            patch_size = 32,
            depth = 12,
            heads = 12,
            dim = 768,
            mlp_dim = 3072,
            dropout = 0.1,
        )
    model = model.train().to(args.device)
    if args.no_compile:
        model = torch.compile(model)
    n_params = count_parameters(model)
    print("model params=", n_params)

    def create_dataloader(split, shuffle):
        dset = get_dataset(
            args,
            split=split,
            transform=transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        )
        return torch.utils.data.DataLoader(
            dset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=shuffle,
        )

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.log_dir, args.name))

    def time_ms():
        t = time.perf_counter_ns()
        return t / 1e6

    optim = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    loss_avg = RollingAvg(args.iter_per_log)
    load_avg = RollingAvg(args.iter_per_log)
    forward_avg = RollingAvg(args.iter_per_log)
    backward_avg = RollingAvg(args.iter_per_log)
    optim_avg = RollingAvg(args.iter_per_log)
    dataloader_avg = RollingAvg(args.iter_per_log)
    total_avg = RollingAvg(args.iter_per_log)

    def log_metrics(val_metrics=None):
        log_t1 = time_ms()
        if val_metrics is None:
            print(f"[{i}] loss={loss:.2f}", flush=True)
        else:
            print(f"[{i}] loss={loss:.2f}, val={val_metrics}", flush=True)
        # TODO: avg of past N iters
        writer.add_scalar("Loss/train", loss_avg.get(), i)
        if val_metrics is not None:
            for k, v in val_metrics.items():
                writer.add_scalar(f"Val/{k}", v, i)

        writer.add_scalar("Perf/dataloader", dataloader_avg.get(), i)
        writer.add_scalar("Perf/forward", forward_avg.get(), i)
        writer.add_scalar("Perf/backward", backward_avg.get(), i)
        writer.add_scalar("Perf/optim", optim_avg.get(), i)
        writer.add_scalar("Perf/total", total_avg.get(), i)
        log_t2 = time_ms()
        writer.add_scalar("Perf/log", log_t2 - log_t1, i)

    train_dloader = create_dataloader("train", True)

    load_t1 = time_ms()
    total_t1 = time_ms()
    i = 0
    while i < args.niter:
        for x in train_dloader:
            load_t2 = time_ms()
            dataloader_avg.add(load_t2 - load_t1)

            forward_t1 = time_ms()
            y = model(x["image"].to(args.device))
            forward_t2 = time_ms()
            forward_avg.add(forward_t2 - forward_t1)

            backward_t1 = time_ms()
            optim.zero_grad()
            target = x["target"].to(args.device)
            loss = F.cross_entropy(y, target)
            loss.backward()
            backward_t2 = time_ms()
            backward_avg.add(backward_t2 - backward_t1)
            loss_avg.add(loss.detach().cpu().item())

            optim_t1 = time_ms()
            optim.step()
            optim_t2 = time_ms()
            optim_avg.add(optim_t2 - optim_t1)

            total_t2 = time_ms()
            total_avg.add(total_t2 - total_t1)
            if (i != 0 or args.val_first_iter) and i % args.val_iter_freq == 0:
                val_dloader = create_dataloader("val", False)
                model = model.eval()
                with torch.no_grad():
                    acc1 = 0
                    acc5 = 0
                    n = 0
                    for x in tqdm(val_dloader):
                        logits = model(x["image"].to(args.device))
                        target = x["target"].to(args.device)
                        a1, a5 = accuracy(logits, target, topk=(1, 5))
                        acc1 += a1
                        acc5 += a5
                        n += x["image"].shape[0]
                    acc1 /= n
                    acc5 /= n
                    log_metrics({"top1": acc1, "top5": acc5})
                model = model.train()
                # TODO: write to checkpoint_dir

            elif i % args.iter_per_log == 0:
                log_metrics()

            i += 1
            if i >= args.niter:
                break

            total_t1 = time_ms()
            load_t1 = time_ms()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ViT',
        description='Trains a ViT',
    )
    parser.add_argument(
        "mode",
        type=str,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        # default=4_096,
        default=256,
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
    )
    parser.add_argument(
        "--iter_per_log",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--val_iter_freq",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ref",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--val_first_iter",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--imagenet_root",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    if args.mode == "test":
        v = ViT(
            image_size = 224,
            patch_size = 16,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            # emb_dropout = 0.1
        )

        v2 = ViTRef(
            image_size = 224,
            patch_size = 16,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        img = torch.randn(1, 3, 224, 224)
        y1 = v(img)
        y2 = v2(img)

        # input = torch.arange(81.).view(1, 9, 9)
        # input = torch.stack([input, input*10, input*100])
        # b, c, h, w = input.shape
        # input.view(b, -1, 3, 3)
        # patches = F.unfold(input, kernel_size=4, stride=4).view(b, -1, patch_dim)
        breakpoint()
    elif args.mode == "create_dset_cache":
        create_dset_cache(args)
    elif args.mode == "check_dataset":
        check_dataset(args)
    else:
        assert args.mode == "train"
        train(args)
