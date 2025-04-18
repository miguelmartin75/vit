# vit

An implementation of ViT in multiple languages and frameworks.
- [x] PyTorch
- [ ] C++ (CPU)
- [ ] C++ CUDA
- [ ] C++ SYCL (Intel GPUs)
- [ ] Mojo

## Feature Matrix

| Lang/Framework | Status      | Quantization           | Optimizer Quantization |
|:---------------|:------------|:-----------------------|:-----------------------|
| PyTorch        | DONE        | TODO                   | TODO                   |
| C++ CPU        | NOT STARTED | TODO                   | TODO                   |
| C++ CUDA       | NOT STARTED | TODO                   | TODO                   |
| Mojo           | NOT STARTED | TODO                   | TODO                   |

## Try it Yourself

See [docs/dev/NOTES.md](./docs/dev/NOTES.md)

# Models

| Model          | Pre-training data   | Top-1 IN-1k            | Link       |
|:---------------|:--------------------|:-----------------------|:-----------|
| ViT-S/32       | IN-1k^*             | 67.8%                  | [gdrive](TODO) |
| ViT-S/32       | IN-21k              | DOING                  | TODO       |

\* note that this is IN-1k, hence the lower Top-1 compared to other pre-trained ViTs.

# Notes

I follow the original [original ViT's](https://github.com/google-research/vision_transformer) hyper-params. This includes:
- LR:
    - Linear warmup of 30 epochs, i.e. increase LR from eps -> lr linearly
    - 
- H-params for the model architecture

## Train/Eval Lessons
1. When training from scratch: 
    1. Warming-up the LR is **very important**. Training a ViT from scratch is not like training a CNN.
    2. Use a large dataset when training from scratch (pre-training), e.g. [IN-21k](https://github.com/Alibaba-MIIL/ImageNet21K)
2. Evaluate with multiple crops ("Inception-style") and average the output logits. I observed a ~+10-15% boost in top-1 performance when doing this.
