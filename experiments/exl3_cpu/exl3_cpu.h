#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
namespace exl3 {
enum class Codebook { mul1, mcg };
enum class Layout { native, band8 };
enum class ISA { scalar, avx512bw, vnni, vbmi };
// Logical tails are zero-padded BEFORE H128; trellis/scales cover padded extents.
struct Matrix {
    size_t k, n, padded_k, padded_n;
    unsigned bits;
    Codebook codebook;
    Layout layout;
    std::vector<uint16_t> trellis;
    std::vector<float> suh, svh;
    std::vector<float> bias; // empty or n logical outputs, applied in FP32 after crop
};
bool supported(ISA isa);
void validate(const Matrix& w);
void repack(Matrix& w, Layout layout);
float codebook_value(uint16_t state, Codebook cb);
void decode_tile(const uint16_t* words, unsigned bits, Codebook cb, ISA isa, float* out);
// Fused numerical path: fp16 codebook, FP32 transforms (no materialized stage rounds).
// Requires a declared error envelope against reconstruct(); not bit-exact canonical output.
void dense(const Matrix& w, const float* x, size_t rows, size_t x_stride,
           float* y, size_t y_stride, ISA isa = ISA::scalar);
// Explicit approximation: affine MUL1 codebook and symmetric Q8 activations.
void mul1_q8(const Matrix& w, const float* x, size_t rows, size_t x_stride,
             float* y, size_t y_stride, ISA isa = ISA::scalar);
// Each route reads input row and writes its own output row (duplicate experts legal).
struct Route { size_t expert, input_row; };
void indexed(const std::vector<Matrix>& experts, const std::vector<Route>& routes,
             const float* x, size_t x_rows, size_t x_stride, float* y, size_t y_stride,
             ISA isa, bool grouped_by_k, bool q8 = false);
// Canonical [k,n] reconstruction; provider accepts row-major A[M,K], B[K,N], C[M,N].
using Gemm = void (*)(size_t, size_t, size_t, const float*, size_t,
                     const float*, float*, size_t);
std::vector<float> reconstruct(const Matrix& w);
void prefill(const Matrix& w, const float* x, size_t rows, size_t x_stride,
             float* y, size_t y_stride, Gemm provider);
}
