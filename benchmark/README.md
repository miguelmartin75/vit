# setup
```
# conda remove -n vit_benchmark --all -y
conda create -n vit_benchmark python=3.12 -y
conda activate vit_benchmark

pip install torch torchvision tqdm tensorboard
```


imagenet
```
python vit_pytorch.py create_dset_cache \
    -nw 20 \
    --dataset_root /datasets01/imagenet_full_size/061417/
    --dataset_name imagenet
```

tinyimagenet
```
python vit_pytorch.py create_dset_cache \
    -nw 20 \
    --dataset_root /datasets01/tinyimagenet/081318/ \
    --dataset_name tinyimagenet
```

# train

```
python vit_pytorch.py train \
    --chkpt_dir /checkpoint/miguelmartin/chkpt/vit_tinyimagenet \
    --name imagenet_bs512 \
    -nw 20 \
    --batch_size 512 \
    --dataset_name imagenet
```

```
python vit_pytorch.py train \
    --chkpt_dir /checkpoint/miguelmartin/chkpt/vit_tinyimagenet \
    --name tinyimagenet_bs512 \
    --num_classes 200 \
    --img_size 64 \
    -nw 20 \
    --batch_size 512 \
    --dataset_name tinyimagenet
```
