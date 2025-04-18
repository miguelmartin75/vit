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
| ViT-B/32       | IN-1k               | 67.8%                  | [gdrive]() |
| ViT-B/32       | IN-21k              | DOING                  | TODO       |

# Train/Eval Notes

1. When training from scratch: 
    1. Warm-up the LR, i.e. increase the learning rate linearly over some number of epochs (30)
    2. Use a large dataset for pre-training, e.g. [IN-21k](https://github.com/Alibaba-MIIL/ImageNet21K)
2. Evaluate with multiple crops ("Inception-style") and average the logits. I observed a ~+10-15% boost in top-1 performance.
