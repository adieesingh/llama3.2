// avx2_kernels.h - AVX2 + FMA matvec kernels for the L3Q1 engine (Llama 3.2).
// Compile with: -O3 -fopenmp -mavx2 -mfma   (or -march=native)
//
// Idea: the scalar loop converts every 4-bit weight to float and multiplies one
// by one. Here we instead
//   1. quantize the activation vector x to int8 once per matvec (per group of 32),
//   2. unpack 32 nibbles into one 256-bit register,
//   3. do 32 int8 multiply-adds in a single maddubs instruction,
//   4. scale the integer result by (weight_scale * x_scale) with one FMA.
// The int8 activation rounding adds a small error (~0.1-0.5%), same trade-off
// llama.cpp makes for its Q4 kernels.
#pragma once

#if defined(__AVX2__) && defined(__FMA__)
#define HAVE_AVX2_KERNELS 1

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

static inline float hsum256_ps(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

// x (float) -> xq (int8) with one float scale per group of 32.
static inline void quantize_x_q8(const float *x, int n, int8_t *xq, float *xs)
{
    const int nb = n / 32;
    for (int g = 0; g < nb; g++)
    {
        const float *xg = x + g * 32;
        float amax = 0.f;
        for (int j = 0; j < 32; j++)
            amax = std::max(amax, std::fabs(xg[j]));
        const float sc = amax / 127.f;
        const float inv = sc > 0.f ? 1.f / sc : 0.f;
        xs[g] = sc;
        for (int j = 0; j < 32; j++)
            xq[g * 32 + j] = (int8_t)lrintf(xg[j] * inv);
    }
}

// 16 packed bytes -> 32 signed int8 weights in range [-8, 7].
// Layout in the file: byte j holds weight j (low nibble) and weight j+16 (high nibble).
static inline __m256i unpack_q4_group(const uint8_t *p)
{
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m128i raw = _mm_loadu_si128((const __m128i *)p);
    const __m128i lo = _mm_and_si128(raw, m0f);                    // weights 0..15
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m0f); // weights 16..31
    const __m256i w = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    return _mm256_sub_epi8(w, _mm256_set1_epi8(8));
}

// acc += scale * dot(wsigned[32], xv[32]) using the abs/sign trick, because
// maddubs needs its first operand unsigned.
static inline __m256 dot_i8_group(__m256i wsigned, __m256i xv, __m256 scale, __m256 acc)
{
    const __m256i ax = _mm256_sign_epi8(wsigned, wsigned); // |w|
    const __m256i sx = _mm256_sign_epi8(xv, wsigned);      // x * sign(w)
    const __m256i p16 = _mm256_maddubs_epi16(ax, sx);      // pairs -> int16
    const __m256i p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(1)); // -> int32 x8
    return _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), scale, acc);
}

static void matvec_q4_avx2(float *y, const float *x, const uint8_t *qbase,
                           const float *sbase, int out, int in)
{
    const int nb = in / 32;
    static thread_local std::vector<int8_t> xq_buf;
    static thread_local std::vector<float> xs_buf;
    xq_buf.resize(in);
    xs_buf.resize(nb);
    quantize_x_q8(x, in, xq_buf.data(), xs_buf.data());
    const int8_t *xq = xq_buf.data();
    const float *xs = xs_buf.data();

#pragma omp parallel for schedule(static)
    for (int o = 0; o < out; o++)
    {
        const uint8_t *q = qbase + (size_t)o * (in / 2);
        const float *s = sbase + (size_t)o * nb;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        int g = 0;
        for (; g + 1 < nb; g += 2) // two accumulators hide FMA latency
        {
            a0 = dot_i8_group(unpack_q4_group(q + g * 16),
                              _mm256_loadu_si256((const __m256i *)(xq + g * 32)),
                              _mm256_set1_ps(s[g] * xs[g]), a0);
            a1 = dot_i8_group(unpack_q4_group(q + (g + 1) * 16),
                              _mm256_loadu_si256((const __m256i *)(xq + (g + 1) * 32)),
                              _mm256_set1_ps(s[g + 1] * xs[g + 1]), a1);
        }
        if (g < nb)
            a0 = dot_i8_group(unpack_q4_group(q + g * 16),
                              _mm256_loadu_si256((const __m256i *)(xq + g * 32)),
                              _mm256_set1_ps(s[g] * xs[g]), a0);
        y[o] = hsum256_ps(_mm256_add_ps(a0, a1));
    }
}

// int8 weights (token embedding / logits matvec, vocab x dim).
static void matvec_q8_avx2(float *y, const float *x, const int8_t *qbase,
                           const float *sbase, int out, int in)
{
    const int nb = in / 32;
    static thread_local std::vector<int8_t> xq_buf;
    static thread_local std::vector<float> xs_buf;
    xq_buf.resize(in);
    xs_buf.resize(nb);
    quantize_x_q8(x, in, xq_buf.data(), xs_buf.data());
    const int8_t *xq = xq_buf.data();
    const float *xs = xs_buf.data();

#pragma omp parallel for schedule(static)
    for (int o = 0; o < out; o++)
    {
        const int8_t *q = qbase + (size_t)o * in;
        const float *s = sbase + (size_t)o * nb;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        int g = 0;
        for (; g + 1 < nb; g += 2)
        {
            a0 = dot_i8_group(_mm256_loadu_si256((const __m256i *)(q + g * 32)),
                              _mm256_loadu_si256((const __m256i *)(xq + g * 32)),
                              _mm256_set1_ps(s[g] * xs[g]), a0);
            a1 = dot_i8_group(_mm256_loadu_si256((const __m256i *)(q + (g + 1) * 32)),
                              _mm256_loadu_si256((const __m256i *)(xq + (g + 1) * 32)),
                              _mm256_set1_ps(s[g + 1] * xs[g + 1]), a1);
        }
        if (g < nb)
            a0 = dot_i8_group(_mm256_loadu_si256((const __m256i *)(q + g * 32)),
                              _mm256_loadu_si256((const __m256i *)(xq + g * 32)),
                              _mm256_set1_ps(s[g] * xs[g]), a0);
        y[o] = hsum256_ps(_mm256_add_ps(a0, a1));
    }
}

#endif // __AVX2__ && __FMA__
