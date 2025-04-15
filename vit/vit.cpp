#include "vit.h"

#include <stdlib.h>
#include <string.h>

#include <fmt/core.h>
#include <fmt/ranges.h>

// https://github.com/fmtlib/fmt/issues/2134
template <typename... T>
void print(fmt::format_string<T...> fmt, T&&... args) {
    fmt::print(fmt, std::forward<T>(args)...);
    fflush(stdout);
}

template <typename... T>
void println(fmt::format_string<T...> fmt, T&&... args) {
    auto str = fmt::format(fmt, std::forward<T>(args)...);
    print("{}\n", str);
}

template<i32 N>
Shape shape_lit(const i32 (&dims)[N]) {
    return Shape{.n = N, .dims = dims};
}


/*
TODO:
- [ ] 
 */

// Tensor rand3(u32 c, u32 h, u32, w) {
//     // TODO
// }
// 
// Tensor rand4(u32 b, u32 c, u32 h, u32, w) {
//     // TODO
// }

// TODO: 
// Tensor operations:
//  - initialization: randn, zeros
//  - concat
//  - slice

i64 shape_nelem(Shape shape) {
    i64 n = shape.n > 0;
    for(int i = 0; i < shape.n; ++i) {
        n *= shape.dims[i];
    }
    return n;
}

void shape_autofill(Shape* shape, Shape ref) {
    // TODO: return errors?
    i64 n = shape_nelem(ref);
    i64 n_fill_idx = -1;
    for(i32 i = 0; i < shape->n; ++i) {
        if(shape->dims[i] == -1) {
            VIT_ASSERT(n_fill_idx == -1, "only 1 dim can be auto-filled");
            n_fill_idx = i;
        } else {
            // TODO: return errors?
            VIT_ASSERT(n % shape->dims[i] == 0, "non-divisble shape");
            n /= shape->dims[i];
        }
    }
    if(n_fill_idx > -1) {
        shape->dims[n_fill_idx] = n;
    }
}

Tensor tensor_view(Tensor x, Shape shape) {
    shape_autofill(&shape, x.shape);
    u64 lhs = shape_nelem(x.shape);
    u64 rhs = shape_nelem(shape);
    // TODO: return errors?
    VIT_ASSERT(lhs == rhs, "shapes incompat");
    VIT_ASSERT(rhs > 0, "shape must not contain negative dims");
    return Tensor{
        // .dtype = x.dtype,
        .data = x.data,
        .shape = shape
    };
}

Tensor tensor_cat(Tensor a, Tensor b, i32 dim) {
    // TODO
}

Tensor tensor_add(Tensor a, Tensor b) {
    // TODO
}

Tensor tensor_mean(Tensor x, i32 dim) {
    // TODO
}

Tensor tensor_variance(Tensor x, i32 dim) {
    // TODO
}

Tensor tensor_layer_norm(Tensor x) {
}

Tensor tensor_zeros(Shape shape) {
    // TODO: use allocator
    i64 nelem = shape_nelem(shape);
    byte8* mem = (byte8*)calloc(sizeof(u32) * shape.n + nelem * sizeof(f32), 1);
    byte8* data = mem + sizeof(u32)*4;
    Shape shape_copy{.n = shape.n, .dims = (i32*)mem};
    memcpy(shape_copy.dims, shape.dims, sizeof(i32) * shape.n);

    return Tensor{
        // TODO device / dtype
        .data = data,
        .shape = shape_copy
    };
}

// CPU
struct ViT_CPU {
    // TODO
    Tensor patch_Wb;
    Tensor class_emb;

    Tensor forward(Tensor input) {
        // TODO: cpu postfix
        Tensor x = tensor_view(input, shape_lit({-1, 16*16*3}));
        x = tensor_layer_norm(x);

        // N x (P*P*C) -> N * D
        x = tensor_matmul(x, ctx->patch_Wb);
        x = tensor_cat(class_emb, x, 1);

        Tensor pos_emb; // TODO: getme
        x = tensor_cat(pos_emb, x, 1);
    }
};


ViT vit_init(ViTModelParams params) {
    // TODO: switch on device
    ViT ret;
    ret.ctx = new ViT_CPU;
    return ret;
}

void vit_destroy(ViT* vit) {
    delete (ViT_CPU*)vit->ctx;
    vit->ctx = nullptr;
}

Tensor vit_forward(ViT* vit, Tensor input) {
    // TODO: batched
    // TODO: moveto vit_forward_cpu
    ViT_CPU* cpu = (ViT_CPU*)vit->ctx;
    cpu->forward(input);
    // 2. [ ] feed into MLP (single layer)
    // 3. [ ] concat with position embeddings
    // 4. [ ] feed into L layers of 

}

Tensor vit_backward(ViT*) {
    // TODO
}

// TODO: moveme to tests
void test_shape() {
    Shape shape = shape_lit({-1, 16, 16});
    VIT_ASSERT(shape_nelem(shape) == -256);

    Shape ref = shape_lit({3, 224, 224});
    shape_autofill(&shape, ref);
    VIT_ASSERT(shape.dims[0] == 588);
}

void test_tensor_init() {
    Tensor zeros = tensor_zeros(shape_lit({3, 224, 224}));
    VIT_ASSERT(shape_nelem(zeros.shape) == 150528);

    Tensor new_view = tensor_view(zeros, shape_lit({-1, 16*16}));
    VIT_ASSERT(shape_nelem(new_view.shape) == 150528);
    VIT_ASSERT(new_view.shape.dims[0] == 588);
}

int main() {
    test_shape();
    test_tensor_init();

    ViT vit = vit_init(ViTModelParams{
        // TODO
        // .device = DEVICE_CPU,
        .patch_size = 16
    });

    Tensor inp = tensor_zeros(shape_lit({3, 224, 224}));
    vit_forward(&vit, inp);
}
