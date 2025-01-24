#include "vit.h"

/*
TODO:
- [ ] 
 */

struct ViT_CPU {
    void forward(Tensor input) {
    }
    Tensor backward() {
    }
};

// CPU

// Tensor rand3(u32 c, u32 h, u32, w) {
//     // TODO
// }
// 
// Tensor rand4(u32 b, u32 c, u32 h, u32, w) {
//     // TODO
// }

Tensor zeros3(u32 c, u32 h, u32, w) {
    // TODO
}

Tensor zeros4(u32 b, u32 c, u32 h, u32, w) {
    // TODO
}

ViT vit_init(ViTModelParams params) {
}
void vit_destroy(ViT*) {
}
void vit_forward(ViT*, Tensor input) {
}
Tensor vit_backward(ViT*) {
}

int main() {
    ViT vit = vit_init(ViTModelParams{
        .device = DEVICE_CPU,
        .patch_size = 16
    });

    Tensor tensor = zeros4(64, 3, 224, 224);
    // vit_forward_cpu();
}
