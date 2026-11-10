// bench_avx2.cpp - scalar vs AVX2 matvec benchmark with Llama 3.2 3B shapes.
//
// Build (Windows MinGW):  g++ -O3 -fopenmp -mavx2 -mfma -static bench_avx2.cpp -o bench_avx2.exe
// Build (Linux/macOS):    g++ -O3 -fopenmp -mavx2 -mfma bench_avx2.cpp -o bench_avx2
// Run:                    .\bench_avx2.exe [--iters 20] [--threads 0] [--no-head]
#include "avx2_kernels.h"

#ifndef HAVE_AVX2_KERNELS
#error "Compile with -mavx2 -mfma (or -march=native)"
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

static const int GS = 32;

// ---- scalar reference: identical to the loops in engine.cpp ----
static void ref_q4(float *y, const float *x, const uint8_t *qbase, const float *sbase, int out, int in)
{
    const int nb = in / GS;
#pragma omp parallel for schedule(static)
    for (int o = 0; o < out; o++)
    {
        const uint8_t *q = qbase + (size_t)o * (in / 2);
        const float *s = sbase + (size_t)o * nb;
        float acc = 0.f;
        for (int g = 0; g < nb; g++)
        {
            const uint8_t *qg = q + g * 16;
            const float *xg = x + g * GS;
            float sum = 0.f;
            for (int j = 0; j < 16; j++)
            {
                int w0 = (int)(qg[j] & 0x0F) - 8;
                int w1 = (int)(qg[j] >> 4) - 8;
                sum += (float)w0 * xg[j] + (float)w1 * xg[j + 16];
            }
            acc += sum * s[g];
        }
        y[o] = acc;
    }
}

static void ref_q8(float *y, const float *x, const int8_t *qbase, const float *sbase, int out, int in)
{
    const int nb = in / GS;
#pragma omp parallel for schedule(static)
    for (int o = 0; o < out; o++)
    {
        const int8_t *q = qbase + (size_t)o * in;
        const float *s = sbase + (size_t)o * nb;
        float acc = 0.f;
        for (int g = 0; g < nb; g++)
        {
            float sum = 0.f;
            for (int j = 0; j < GS; j++)
                sum += (float)q[g * GS + j] * x[g * GS + j];
            acc += sum * s[g];
        }
        y[o] = acc;
    }
}

struct Shape
{
    const char *name;
    int out, in;
    bool q8;
    int per_token; // how many times this shape runs per generated token (3B: 28 layers)
};

struct Weights
{
    std::vector<uint8_t> q;
    std::vector<float> s;
};

static double now_ms()
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int main(int argc, char **argv)
{
    int iters = 20, threads = 0;
    bool no_head = false;
    for (int i = 1; i < argc; i++)
    {
        std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc)
            iters = atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc)
            threads = atoi(argv[++i]);
        else if (a == "--no-head")
            no_head = true;
    }
#ifdef _OPENMP
    if (threads > 0)
        omp_set_num_threads(threads);
    printf("OpenMP threads: %d\n", omp_get_max_threads());
#endif
    printf("Iterations per kernel: %d (plus 2 warm-up)\n\n", iters);

    // Llama 3.2 3B: dim 3072, hidden 8192, 28 layers, q_dim 3072, kv_dim 1024, vocab 128256
    const Shape shapes[] = {
        {"wq/wo  3072x3072", 3072, 3072, false, 28 * 2},
        {"wk/wv  1024x3072", 1024, 3072, false, 28 * 2},
        {"wg/wu  8192x3072", 8192, 3072, false, 28 * 2},
        {"wd     3072x8192", 3072, 8192, false, 28 * 1},
        {"logits 128256x3072 (q8)", 128256, 3072, true, 1},
    };
    const int COPIES = 3; // rotate through copies so the 8 MB L3 cannot hold everything

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> sd(0.001f, 0.02f);

    double tot_ref = 0, tot_avx = 0;

    printf("%-26s %10s %10s %8s %9s %10s\n", "matrix", "scalar ms", "avx2 ms", "speedup", "avx2 GB/s", "rel.err");
    for (const Shape &sh : shapes)
    {
        if (sh.q8 && no_head)
            continue;
        const int nb = sh.in / GS;
        const size_t qbytes = sh.q8 ? (size_t)sh.out * sh.in : (size_t)sh.out * sh.in / 2;
        const size_t sn = (size_t)sh.out * nb;
        const int copies = sh.q8 ? 1 : COPIES;

        std::vector<Weights> W(copies);
        for (auto &w : W)
        {
            w.q.resize(qbytes);
            w.s.resize(sn);
            uint32_t *p = (uint32_t *)w.q.data();
            for (size_t i = 0; i < qbytes / 4; i++)
                p[i] = (uint32_t)rng();
            if (sh.q8) // keep int8 in [-127,127] like the converter does
                for (size_t i = 0; i < qbytes; i++)
                    if ((int8_t)w.q[i] == -128)
                        w.q[i] = (uint8_t)(int8_t)-127;
            for (size_t i = 0; i < sn; i++)
                w.s[i] = sd(rng);
        }
        std::vector<float> x(sh.in), yr(sh.out), ya(sh.out);
        for (auto &v : x)
            v = nd(rng);

        auto run_ref = [&](int i)
        {
            const Weights &w = W[i % copies];
            if (sh.q8)
                ref_q8(yr.data(), x.data(), (const int8_t *)w.q.data(), w.s.data(), sh.out, sh.in);
            else
                ref_q4(yr.data(), x.data(), w.q.data(), w.s.data(), sh.out, sh.in);
        };
        auto run_avx = [&](int i)
        {
            const Weights &w = W[i % copies];
            if (sh.q8)
                matvec_q8_avx2(ya.data(), x.data(), (const int8_t *)w.q.data(), w.s.data(), sh.out, sh.in);
            else
                matvec_q4_avx2(ya.data(), x.data(), w.q.data(), w.s.data(), sh.out, sh.in);
        };

        // correctness on copy 0
        run_ref(0);
        run_avx(0);
        double num = 0, den = 0;
        for (int i = 0; i < sh.out; i++)
        {
            double d = (double)ya[i] - yr[i];
            num += d * d;
            den += (double)yr[i] * yr[i];
        }
        const double rel = den > 0 ? std::sqrt(num / den) : 0;

        for (int i = 0; i < 2; i++) { run_ref(i); run_avx(i); } // warm-up
        double t0 = now_ms();
        for (int i = 0; i < iters; i++) run_ref(i);
        const double ms_ref = (now_ms() - t0) / iters;
        t0 = now_ms();
        for (int i = 0; i < iters; i++) run_avx(i);
        const double ms_avx = (now_ms() - t0) / iters;

        const double gbs = (double)(qbytes + sn * 4) / (ms_avx * 1e6);
        printf("%-26s %10.3f %10.3f %7.2fx %9.1f %9.4f%%\n", sh.name, ms_ref, ms_avx, ms_ref / ms_avx, gbs, rel * 100);
        tot_ref += ms_ref * sh.per_token;
        tot_avx += ms_avx * sh.per_token;
    }

    printf("\nTime for all matvecs of one token%s: scalar %.0f ms, AVX2 %.0f ms\n",
           no_head ? " (without logits)" : "", tot_ref, tot_avx);
    printf("Matvec-only upper bound on decode speed: scalar %.2f tok/s -> AVX2 %.2f tok/s\n",
           1000.0 / tot_ref, 1000.0 / tot_avx);
    printf("(Real speed is lower: attention, RoPE, norms and sampling are not counted.)\n");
    return 0;
}
