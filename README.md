# vit

An implementation of ViT in multiple languages and frameworks.
- [x] PyTorch
- [ ] C++ (CPU)
- [ ] C++ CUDA
- [ ] C++ SYCL (Intel GPUs)
- [ ] Mojo

>[!NOTE]
>PyTorch is the only implementation I have finished. Other implementations are a
work-in-progress.

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
| ViT-S/32       | IN-1k               | 67.8%*                 | [gdrive](https://drive.google.com/file/d/1ACvPXIMwKPOwP2ijTeXO5rb0GBNyqmIj/view?usp=sharing) |
| ViT-S/32       | IN-21k              | DOING                  | TODO       |

\* yes this model gets out performed by a ResNet-50 on IN-1k. However, this
uses IN-1k as it's pre-training dataset, hence the low Top-1 for a ViT. ViTs
require a ~~large~~ huge pre-training dataset to obtain good performance (due
to no inductive bias on images, unlike CNNs).

# Notes

I followed the [original ViT's](https://github.com/google-research/vision_transformer) hyper-params. This includes:
- LR:
    - Linear warmup of 30 epochs, i.e. increase LR from eps -> lr linearly
    - Cosine decay to 0 for the remaining epochs. Note, the hparams for cosine decay are for half a period of cosine, i.e. it doesn't go back up to original LR.
- H-params for the model architecture

## Train/Eval Lessons
Training a ViT from scratch is not easy (due to compute requirements and hparam tweaking).

1. When training from scratch: 
    1. Warming-up the LR is **very important**. Training a ViT from scratch is not like training a CNN.
    2. Use a huge dataset when training from scratch (pre-training), e.g. [IN-21k](https://github.com/Alibaba-MIIL/ImageNet21K)
2. Evaluate with multiple crops ("Inception-style") and average the output logits. I observed a ~+10-15% boost in top-1 performance when doing this.
