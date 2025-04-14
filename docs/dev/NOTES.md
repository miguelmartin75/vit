# setup

## python

### CPU or Nvidia GPU
```
export ENV_NAME=py312
# conda remove -n vit_benchmark --all -y
conda create -n $ENV_NAME python=3.12 -y
conda activate $ENV_NAME

pip install torch torchvision tqdm tensorboard einops
```

### Intel GPU

https://pytorch-extension.intel.com/installation?platform=gpu&version=v2.5.10%2Bxpu&os=linux%2Fwsl2&package=pip
```
# https://github.com/intel/intel-extension-for-pytorch/issues/702
conda install -c conda-forge libstdcxx-ng
python -m pip install torch==2.3.1+cxx11.abi torchvision==0.18.1+cxx11.abi torchaudio==2.3.1+cxx11.abi intel-extension-for-pytorch==2.3.110+xpu oneccl_bind_pt==2.3.100+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

# datasets

## imagenet
```
python vit/pt.py create_dset_cache \
    -nw 20 \
    --dataset_root /datasets/imagenet/
    --dataset_name imagenet
```

# train

## pytorch
```
python vit/pt.py train \
    --chkpt_dir /checkpoint/$USER/vit/ \
    --img_size 224 \
    --name vit-s/32_ep-300_lr-0.001_do-0.00_wd-0.03 \
    --model_template vit-s \
    --patch_size 32 \
    --dropout 0.1 \
    -nw 20 \
    --batch_size 512 \
    --warmup_epochs 30 \
    --epochs 300 \
    --weight_decay 0.03 \
    --lr 0.001 \
    --betas 0.9 0.999 \
    --val_iter_freq 1500 \
    --dataset_name imagenet
```

# TODOs

- [ ] PyTorch
    - [ ] evaluate with multiple crops and avg.
    - [ ] continue training
