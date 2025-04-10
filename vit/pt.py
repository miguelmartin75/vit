import math
import gc
import copy
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

DSET_CACHE_DIR = "./datasets/"
MODEL_TEMPLATES = {
    "vit-s": {
        "depth" : 12,
        "heads" : 6,
        "dim" : 384,
        "mlp_dim" : 1536,
    },
    "vit-b": {
        "depth" : 12,
        "heads" : 12,
        "dim" : 768,
        "mlp_dim" : 3072,
    }
}

# * SECTION: utilities
def time_ms():
    t = time.perf_counter_ns()
    return t / 1e6

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
        if not self.data:
            return 0
        return sum(self.data) / len(self.data)

def list_files(path):
    for root, _, files in os.walk(path):
        for f in files:
            yield os.path.join(root, f)

# TODO: simplify this
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

# * SECTION: model
class MSA(nn.Module):
    def __init__(self, dim, heads, dropout, fused = False):
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
    def __init__(self, dim, heads, mlp_dim, dropout):
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
    def __init__(self, patch_size=16, mlp_dim=2048, dim=1024, heads=8, depth=8, dropout=0.0, num_classes=1000, image_size=224, channels=3):
        super().__init__()

        self.patch_size = patch_size
        self.dim = dim
        self.c = channels
        self.patch_dim = patch_size * patch_size * self.c
        self.N = (image_size*image_size) // (patch_size*patch_size)

        # NOTE: zeros
        # https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py#L281
        # self.class_tok = nn.Parameter(torch.randn(1, 1, dim))
        self.class_tok = nn.Parameter(torch.zeros(1, 1, dim))
        # NOTE:
        # TODO: * 0.02?
        self.pos_emb = nn.Parameter(torch.randn(1, self.N + 1, dim))

        # TODO: check GELU not needed here
        self.patch_emb = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
            nn.LayerNorm(dim),  # TODO: note redundant layer norm
        )
        # NOTE: https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py#L306-L307
        self.classify_head = nn.Linear(dim, num_classes)
        self.dropout = nn.Dropout(dropout)

        self.transformer = TransformerEncoder(
            n_layers=depth,
            heads=heads,
            dim=dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.reset_parameters()

    def reset_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # NOTE
                # https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L118-L122
                # in ViT: 
                # https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py#L74-L77
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.normal_(1e-6)
            # elif isinstance(m, nn.LayerNorm):
            #     m.weight.data.fill_(0)
            #     m.bias.data.fill_(0)
        self.apply(init_weights)

        with torch.no_grad():
            self.classify_head.weight.fill_(0)
            self.classify_head.bias.fill_(0)

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
        x += self.pos_emb
        x = self.dropout(x)

        x = self.transformer(x)

        x = x[:, 0]
        x = self.classify_head(x)

        return x

# * SECTION: dataloading
def _load_img(x):
    if ".JPEG" not in x:
        return None

    if not os.path.exists(x[".JPEG"]):
        return None
    try:
        _ = Image.open(x[".JPEG"]).convert("RGB")
    except:
        print(traceback.format_exc())
        return None
    return x

def create_imagenet_dset_cache(args, split, shuffle, name, labels=None):
    root = args.dataset_root
    if labels is None:
        label_txt_path = os.path.join(root, "labels.txt")
        if os.path.exists(label_txt_path):
            labels = [(x.split(","), i) for i, x in enumerate(open(label_txt_path).readlines())]
            labels = {x[0]: (i, x[1].strip()) for x, i in labels}
        else:
            labels = {}

    cache_path = os.path.join(DSET_CACHE_DIR, f"{name}-{split}.jsonl")
    os.makedirs(DSET_CACHE_DIR, exist_ok=True)

    data = list(dataset_folder_iter(
        os.path.join(root, split),
        0,
        None,
        shuffle=False,
    ))

    map_fn = _load_img
    with open(cache_path, "w") as out_f:
        print("Writing cache...")
        count = 0
        xs = []
        id = 0
        for x in local_map(data, map_fn, num_workers=args.num_workers, progress=True):
            if x is None:
                continue
            key = x["dir"].split("/")[0]
            if key not in labels:
                print(f"WARN: adding, {key} to labels")
                labels[key] = (id, key)
                id += 1
            x["target"] = labels[key][0]
            xs.append(x)

        if shuffle:
            random.seed(42)
            random.shuffle(xs)

        for x in tqdm(xs):
            out_f.write(json.dumps(x) + "\n")
            count += 1
        print(f"{len(data)} -> {count}", flush=True)
    return labels


def create_dset_cache(args):
    labels = create_imagenet_dset_cache(args, "train", True, name=args.dataset_name)
    create_imagenet_dset_cache(args, "val", False, name=args.dataset_name, labels=labels)


def load_sample(x, transform):
    with Image.open(x[".JPEG"]) as img:
        img = img.convert("RGB")
        x["image"] = img
    if transform is not None:
        x["image"] = transform(x["image"])
    return x

class JsonlDset(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        super().__init__()
        self.transform = transform
        self.path = path
        self.data = [json.loads(x) for x in open(path).readlines()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return load_sample(copy.deepcopy(self.data[idx]), self.transform)


def get_dataset(args, split, transform, shuffle):
    if args.mnist:
        assert split in ("train", "val")
        return torchvision.datasets.MNIST(
            root=DSET_CACHE_DIR,
            train=(split=="train"),
            download=True,
            transform=transform,
        )

    if args.use_iter_dsets:
        from vit.vit_pt_ext import JsonlDsetIter
        return JsonlDsetIter(
            path=os.path.join(DSET_CACHE_DIR, f"{args.dataset_name}-{split}.jsonl"),
            transform=transform,
            shuffle=shuffle,
            shuffle_buf_size=args.shuffle_buf_size,
        )
    else:
        return JsonlDset(
            path=os.path.join(DSET_CACHE_DIR, f"{args.dataset_name}-{split}.jsonl"),
            transform=transform,
        )

def create_dataloader(args, split, shuffle, device):
    prefix_transforms = [
        transforms.Resize((args.img_size, args.img_size)),
    ]
    if split == "train":
        prefix_transforms = [
            transforms.Resize(int(2*args.img_size)),
            transforms.RandomCrop(int(args.img_size*1.5)),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5, hue=0.3),
            transforms.RandomGrayscale(p=0.05),
        ]

    dset = get_dataset(
        args,
        split=split,
        transform=transforms.Compose(prefix_transforms + [
            transforms.ToTensor(),
            # [0, 1] -> [-1, -1]
            transforms.Lambda(lambda x: (x-0.5)/0.5),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        shuffle=shuffle,
    )
    dloader_kwargs = {
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "shuffle": False if args.use_iter_dsets else shuffle,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.num_workers > 0,
        "pin_memory_device": device if args.pin_memory else "",
    }
    return torch.utils.data.DataLoader(
        dset,
        **dloader_kwargs,
    )

def profile_dataloader(args):
    assert args.profile_nsamples > args.profile_warmup, "need at least 1 sample to measure"

    dloader = create_dataloader(args, "train", False, args.device)
    device = args.device

    N = len(dloader) if args.profile_nsamples < 0 else args.profile_nsamples
    print(f"profiling {N} batches, device={device}, warmup={args.profile_warmup}")

    t_samples = []
    i = 0
    n_samples = 0
    t1 = time_ms()
    for x in tqdm(dloader, total=N):
        img = x["image"].to(device)
        target = x["target"].to(device)
        n_samples += img.shape[0]
        t2 = time_ms()
        del img, target

        t_samples.append(t2 - t1)
        i += 1
        if i > N:
            break

        t1 = time_ms()

    t_samples = torch.tensor(t_samples[args.profile_warmup:])
    print(f"mean={t_samples.mean():.3f}ms, std={t_samples.std():.3f}ms, min={t_samples.min():.3f}ms, max={t_samples.max():.3f}ms", flush=True)
    print(f"samples/sec={1000*n_samples / t_samples.sum():.3f}")

def create_lr_step_fn(args):
    base_lr = args.lr

    def lr_step_fn(i):
        if args.warmup_iter and i < args.warmup_iter:
            return ((i+1) / args.warmup_iter) * base_lr
        else:
            # cosine schedule
            # https://github.com/google-research/vision_transformer/blob/main/vit_jax/utils.py#L87

            t = torch.tensor((i - args.warmup_iter) / (args.niter - args.warmup_iter))
            t = torch.clamp(t, 0.0, 1.0)
            lr = base_lr * 0.5 * (1 + torch.cos(torch.pi * t))
            return lr

    return lr_step_fn

def train(args):
    # TODO: continue from checkpoint
    assert args.chkpt_dir is not None, "--chkpt_dir not provided"
    assert args.name is not None, "--name not provided"
    assert args.model_template in MODEL_TEMPLATES, f"unknown model template: {args.model_template}"

    model = ViT(
        image_size = args.img_size,
        num_classes = args.num_classes,
        patch_size = args.patch_size,
        dropout = args.dropout,
        channels = 3 if not args.mnist else 1,
        **MODEL_TEMPLATES[args.model_template],
    )

    dtype = torch.float32
    # TODO: does it learn?
    if args.bfloat16:
        dtype = torch.bfloat16

    model = model.train().to(args.device, dtype=dtype)
    if args.no_compile:
        model = torch.compile(model)
    n_params = count_parameters(model)
    print("model params=", n_params)

    chkpt_dir = os.path.join(args.chkpt_dir, args.name)
    log_dir = os.path.join(args.log_dir, args.name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(chkpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    device = args.device
    # optim = torch.optim.Adam(
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=args.betas,
        weight_decay=args.weight_decay,
    )
    loss_avg = RollingAvg(args.iter_per_log)
    load_avg = RollingAvg(args.iter_per_log)
    forward_avg = RollingAvg(args.iter_per_log)
    backward_avg = RollingAvg(args.iter_per_log)
    optim_avg = RollingAvg(args.iter_per_log)
    dataloader_avg = RollingAvg(args.iter_per_log)
    total_avg = RollingAvg(args.iter_per_log)

    i = 0
    def log_metrics(val_metrics=None):
        log_t1 = time_ms()
        if val_metrics is None:
            print(f"[{i}, {i/len(train_dloader):.1f}epochs] loss={loss:.2f}", flush=True)
        else:
            print(f"[{i}, {i/len(train_dloader):.1f}epochs] loss={loss:.2f} val={val_metrics}", flush=True)
        # TODO: avg of past N iters
        writer.add_scalar("Loss/train", loss_avg.get(), i)
        writer.add_scalar("lr", lr, i)
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

    train_dloader = create_dataloader(args, "train", shuffle=True, device=device)
    val_dloader = create_dataloader(args, "val", shuffle=False, device=device)
    if args.niter is None:
        assert args.epochs is not None
        args.niter = len(train_dloader) * args.epochs
    if args.warmup_iter is None:
        assert args.warmup_epochs is not None
        args.warmup_iter = len(train_dloader) * args.warmup_epochs

    print("niter=", args.niter, flush=True)
    print("warmup niter=", args.warmup_iter, flush=True)
    load_t1 = time_ms()
    total_t1 = time_ms()
    last_k_chkpts = {}

    lr_fn = create_lr_step_fn(args)
    while i < args.niter:
        for x in train_dloader:
            load_t2 = time_ms()
            dataloader_avg.add(load_t2 - load_t1)

            lr = lr_fn(i)
            for param_group in optim.param_groups:
                if isinstance(param_group["lr"], torch.Tensor):
                    param_group["lr"].fill_(lr)
                else:
                    param_group["lr"] = lr

            if args.mnist:
                # TODO: deleteme by fixing
                x = {"image": x[0], "target": x[1]}

            forward_t1 = time_ms()

            img = x["image"].to(device, dtype=dtype)
            y = model(img)

            forward_t2 = time_ms()
            forward_avg.add(forward_t2 - forward_t1)

            backward_t1 = time_ms()

            optim.zero_grad()
            target = x["target"].to(device)
            loss = F.cross_entropy(y, target)
            loss.backward()

            backward_t2 = time_ms()
            backward_avg.add(backward_t2 - backward_t1)
            loss_avg.add(loss.detach().cpu().item())

            optim_t1 = time_ms()
            if args.gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

            optim.step()
            optim_t2 = time_ms()
            optim_avg.add(optim_t2 - optim_t1)

            total_t2 = time_ms()
            total_avg.add(total_t2 - total_t1)
            del img, target
            if (i != 0 or args.val_first_iter) and i % args.val_iter_freq == 0:
                model = model.eval()
                with torch.no_grad():
                    acc1 = 0
                    acc5 = 0
                    n = 0
                    for x in tqdm(val_dloader):
                        if args.mnist:  # TODO: fixme
                            x = {"image": x[0], "target": x[1]}

                        img = x["image"].to(device)
                        logits = model(img)
                        logits = F.softmax(logits, dim=-1)
                        target = x["target"].to(device)
                        a1, a5 = accuracy(logits, target, topk=(1, 5))
                        acc1 += a1
                        acc5 += a5
                        n += x["image"].shape[0]
                        del logits, img, target

                    acc1 /= n
                    acc5 /= n
                    top1 = acc1 * 100
                    top5 = acc5 * 100
                    log_metrics({"top1": top1, "top5": top5})

                should_save = False
                if len(last_k_chkpts) >= args.keep_k_chkpts:
                    for idx, info in last_k_chkpts.items():
                        if info["top1"] < top1:
                            should_save = True
                            os.remove(info["path"])
                            del last_k_chkpts[idx]
                            break
                else:
                    should_save = True

                if should_save:
                    to_save = {
                        "model_state_dict": model.state_dict(), 
                        "optim_state_dict": optim.state_dict(), 
                        "top1": top1, "top5": top5,
                        "iter": i,
                        "loss": loss_avg.get(),
                    }
                    path = os.path.join(chkpt_dir, f"{i}-top1:{top1:.1f}.pt")
                    torch.save(to_save, path)
                    last_k_chkpts[i] = {}
                    last_k_chkpts[i]["top1"] = top1
                    last_k_chkpts[i]["path"] = path

                model = model.train()

            elif i % args.iter_per_log == 0:
                log_metrics()

            i += 1

            if i % 1000 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            if i >= args.niter:
                break

            total_t1 = time_ms()
            load_t1 = time_ms()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ViT',
        description='Trains a ViT using PyTorch',
    )
    parser.add_argument(
        "mode",
        type=str,
    )
    parser.add_argument(
        "--model_template",
        type=str,
        default="vit-s",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs='+',
        default=[0.9, 0.999],
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--shuffle_buf_size",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--use_iter_dsets",
        action="store_true",
        default=False,
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
        default=None,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--profile_nsamples",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--warmup_iter",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--profile_warmup",
        type=int,
        default=15,
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
        "--patch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--val_iter_freq",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--chkpt_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--keep_k_chkpts",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=1.0,
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
        "--pin_memory",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="imagenet",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--mnist",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        # TODO: generalize?
        "--bfloat16",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
    )
    args = parser.parse_args()

    if args.mode == "create_dset_cache":
        create_dset_cache(args)
    elif args.mode == "profile_dataloader":
        profile_dataloader(args)
    else:
        assert args.mode == "train"
        train(args)
