#pragma once

#include <stdint.h>

#ifdef  __cplusplus
extern "C" {
#endif

#define VIT_INTERNAL static
#define VIT_ASSERT(x, ...) if(x) __builtin_debugtrap()

typedef uint8_t byte8;
typedef float f32;
typedef double f64;
typedef int32_t i32;
typedef int64_t i64;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int64_t ll;
typedef int32_t rune;


typedef enum {
    DEVICE_CPU,
    DEVICE_CUDA,
    // TODO SYLC
} Device;

typedef struct {
    Device device;
    int patch_size;
} ViTModelParams;

typedef struct {
    Device device;
    void* data;
} ViT;

typedef struct {
    float* data;
    uint32_t* shape;
} Tensor;

ViT vit_init(ViTModelParams params);
void vit_destroy(ViT*);
void vit_forward(ViT*, Tensor input);
Tensor vit_backward(ViT*);

#ifdef  __cplusplus
}
#endif

