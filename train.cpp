#include "vit.h"

int main() {
    ViT vit = vit_init(ViTModelParams{
        // TODO
        // .device = DEVICE_CPU,
        .patch_size = 16
    });

    Tensor inp = tensor_zeros(shape_lit({3, 224, 224}));
    vit_forward(&vit, inp);
}
