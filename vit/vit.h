#pragma once

#include <stdint.h>

#ifdef  __cplusplus
extern "C" {
#endif

// TODO: move to _internal ?
#define VIT_INTERNAL static
// #define VIT_ASSERT(x, ...) if(x) __builtin_debugtrap()
#define VIT_ASSERT(x, ...) if(!(x)) __builtin_trap()

#define countof(a)    (u64)(sizeof(a) / sizeof(*(a)))
#define lengthof(s)   (countof(s) - 1)

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
} DeviceKind;

typedef struct {
    DeviceKind kind : 8;
    u32 idx : 24;
} Device;

typedef struct {
    // TODO
    // Device device;
    int patch_size;
} ViTModelParams;

typedef struct {
    // TODO
    // Device device;
    void* ctx;
} ViT;

typedef struct {
    i32 rw; // read-only
    i32 n;
    i32* dims;
} Shape;

typedef struct {
    // TODO
    // Device device;
    // DType dtype;
    byte8* data;
    Shape shape;
} Tensor;

ViT vit_init(ViTModelParams params);
void vit_destroy(ViT*);
void vit_forward(ViT*, Tensor input);
Tensor vit_backward(ViT*);


#ifdef  __cplusplus
}
#endif

#include <algorithm>

template<i32 N>
Shape shape_lit(const i32 (&dims)[N]) {
    // TODO: allocator
    i32* data = new i32[N];
    // TODO: remove <algorithm> usage
    std::copy(dims, dims + N, data);
    return Shape{.rw = 0, .n = N, .dims = data};
}

#if 1  // TODO: move to impl?

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

#endif
