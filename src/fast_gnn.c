#include "fast_gnn.h"

#include <stdlib.h>
#include <immintrin.h>

#define DIM 16
#define RM 6

#define PREFETCH 32

typedef float v8sf __attribute__((vector_size(32)));

static inline void message_kernel(uint32_t u,
                                  uint32_t N, const uint32_t *V, const uint32_t *E,
                                  const float *x, float *y)
{
    v8sf y0 = _mm256_setzero_ps();
    v8sf y1 = _mm256_setzero_ps();
    v8sf y2 = _mm256_load_ps(x + u * DIM);
    v8sf y3 = _mm256_load_ps(x + u * DIM + 8);

    for (uint32_t i = V[u]; i < V[u + 1]; i++)
    {
        if (i + PREFETCH < V[N])
            __builtin_prefetch(x + E[i + PREFETCH] * DIM);

        uint32_t v = E[i];
        v8sf v0 = _mm256_load_ps(x + v * DIM);
        v8sf v1 = _mm256_load_ps(x + v * DIM + 8);

        y0 = _mm256_add_ps(y0, v0);
        y1 = _mm256_add_ps(y1, v1);
    }

    float deg = V[u + 1] - V[u];
    if (deg > 0.0f)
    {
        v8sf d = _mm256_set1_ps(deg);
        y0 = _mm256_div_ps(y0, d);
        y1 = _mm256_div_ps(y1, d);
    }

    _mm256_store_ps(y, y0);
    _mm256_store_ps(y + 8, y1);
    _mm256_store_ps(y + 16, y2);
    _mm256_store_ps(y + 24, y3);
}

static inline void message_kernel_input(uint32_t u,
                                        uint32_t N, const uint32_t *V, const uint32_t *E, int md,
                                        float *y)
{
    v8sf y0 = _mm256_setzero_ps();
    v8sf y1 = _mm256_setzero_ps();

    v8sf y2 = _mm256_set1_ps(V[u + 1] - V[u]);
    v8sf y3 = _mm256_set1_ps((double)u / (double)N);

    v8sf vmd = _mm256_set1_ps(md);
    y2 = _mm256_div_ps(y2, vmd);

    for (uint32_t i = V[u]; i < V[u + 1]; i++)
    {
        if (i + PREFETCH < V[N])
            __builtin_prefetch(V + E[i + PREFETCH]);

        uint32_t v = E[i];

        v8sf v0 = _mm256_set1_ps(V[v + 1] - V[v]);
        v0 = _mm256_div_ps(v0, vmd);
        v8sf v1 = _mm256_set1_ps((double)v / (double)N);

        y0 = _mm256_add_ps(y0, v0);
        y1 = _mm256_add_ps(y1, v1);
    }

    float deg = V[u + 1] - V[u];
    if (deg > 0.0f)
    {
        v8sf d = _mm256_set1_ps(deg);
        y0 = _mm256_div_ps(y0, d);
        y1 = _mm256_div_ps(y1, d);
    }

    _mm256_store_ps(y, y0);
    _mm256_store_ps(y + 8, y1);
    _mm256_store_ps(y + 16, y2);
    _mm256_store_ps(y + 24, y3);
}

static inline void dense_kernel(const float *A, const float *B, const float *bias, float *C)
{
    v8sf c00 = _mm256_load_ps(bias);
    v8sf c01 = _mm256_load_ps(bias + 8);
    v8sf c10 = _mm256_load_ps(bias);
    v8sf c11 = _mm256_load_ps(bias + 8);
    v8sf c20 = _mm256_load_ps(bias);
    v8sf c21 = _mm256_load_ps(bias + 8);
    v8sf c30 = _mm256_load_ps(bias);
    v8sf c31 = _mm256_load_ps(bias + 8);
    v8sf c40 = _mm256_load_ps(bias);
    v8sf c41 = _mm256_load_ps(bias + 8);
    v8sf c50 = _mm256_load_ps(bias);
    v8sf c51 = _mm256_load_ps(bias + 8);

#pragma GCC unroll 8
    for (int i = 0; i < DIM * 2; i++)
    {
        v8sf b0 = _mm256_load_ps(B + i * DIM);
        v8sf b1 = _mm256_load_ps(B + i * DIM + 8);

        v8sf a0 = _mm256_broadcast_ss(A + i);
        c00 = _mm256_fmadd_ps(b0, a0, c00);
        c01 = _mm256_fmadd_ps(b1, a0, c01);

        v8sf a1 = _mm256_broadcast_ss(A + (DIM * 2) + i);
        c10 = _mm256_fmadd_ps(b0, a1, c10);
        c11 = _mm256_fmadd_ps(b1, a1, c11);

        v8sf a2 = _mm256_broadcast_ss(A + 2 * (DIM * 2) + i);
        c20 = _mm256_fmadd_ps(b0, a2, c20);
        c21 = _mm256_fmadd_ps(b1, a2, c21);

        v8sf a3 = _mm256_broadcast_ss(A + 3 * (DIM * 2) + i);
        c30 = _mm256_fmadd_ps(b0, a3, c30);
        c31 = _mm256_fmadd_ps(b1, a3, c31);

        v8sf a4 = _mm256_broadcast_ss(A + 4 * (DIM * 2) + i);
        c40 = _mm256_fmadd_ps(b0, a4, c40);
        c41 = _mm256_fmadd_ps(b1, a4, c41);

        v8sf a5 = _mm256_broadcast_ss(A + 5 * (DIM * 2) + i);
        c50 = _mm256_fmadd_ps(b0, a5, c50);
        c51 = _mm256_fmadd_ps(b1, a5, c51);
    }

    c00 = _mm256_max_ps(c00, _mm256_setzero_ps());
    c01 = _mm256_max_ps(c01, _mm256_setzero_ps());
    c10 = _mm256_max_ps(c10, _mm256_setzero_ps());
    c11 = _mm256_max_ps(c11, _mm256_setzero_ps());
    c20 = _mm256_max_ps(c20, _mm256_setzero_ps());
    c21 = _mm256_max_ps(c21, _mm256_setzero_ps());
    c30 = _mm256_max_ps(c30, _mm256_setzero_ps());
    c31 = _mm256_max_ps(c31, _mm256_setzero_ps());
    c40 = _mm256_max_ps(c40, _mm256_setzero_ps());
    c41 = _mm256_max_ps(c41, _mm256_setzero_ps());
    c50 = _mm256_max_ps(c50, _mm256_setzero_ps());
    c51 = _mm256_max_ps(c51, _mm256_setzero_ps());

    _mm256_stream_ps(C, c00);
    _mm256_stream_ps(C + 8, c01);
    _mm256_stream_ps(C + 16, c10);
    _mm256_stream_ps(C + 24, c11);
    _mm256_stream_ps(C + 32, c20);
    _mm256_stream_ps(C + 40, c21);
    _mm256_stream_ps(C + 48, c30);
    _mm256_stream_ps(C + 56, c31);
    _mm256_stream_ps(C + 64, c40);
    _mm256_stream_ps(C + 72, c41);
    _mm256_stream_ps(C + 80, c50);
    _mm256_stream_ps(C + 88, c51);
}

static inline void dense_kernel_output(const float *A, const float *B, const float *bias, double *C)
{
    v8sf c00 = _mm256_load_ps(bias);
    v8sf c01 = _mm256_load_ps(bias + 8);
    v8sf c10 = _mm256_load_ps(bias);
    v8sf c11 = _mm256_load_ps(bias + 8);
    v8sf c20 = _mm256_load_ps(bias);
    v8sf c21 = _mm256_load_ps(bias + 8);
    v8sf c30 = _mm256_load_ps(bias);
    v8sf c31 = _mm256_load_ps(bias + 8);
    v8sf c40 = _mm256_load_ps(bias);
    v8sf c41 = _mm256_load_ps(bias + 8);
    v8sf c50 = _mm256_load_ps(bias);
    v8sf c51 = _mm256_load_ps(bias + 8);

#pragma GCC unroll 8
    for (int i = 0; i < DIM * 2; i++)
    {
        v8sf b0 = _mm256_load_ps(B + i * DIM);
        v8sf b1 = _mm256_load_ps(B + i * DIM + 8);

        v8sf a0 = _mm256_broadcast_ss(A + i);
        c00 = _mm256_fmadd_ps(b0, a0, c00);
        c01 = _mm256_fmadd_ps(b1, a0, c01);

        v8sf a1 = _mm256_broadcast_ss(A + (DIM * 2) + i);
        c10 = _mm256_fmadd_ps(b0, a1, c10);
        c11 = _mm256_fmadd_ps(b1, a1, c11);

        v8sf a2 = _mm256_broadcast_ss(A + 2 * (DIM * 2) + i);
        c20 = _mm256_fmadd_ps(b0, a2, c20);
        c21 = _mm256_fmadd_ps(b1, a2, c21);

        v8sf a3 = _mm256_broadcast_ss(A + 3 * (DIM * 2) + i);
        c30 = _mm256_fmadd_ps(b0, a3, c30);
        c31 = _mm256_fmadd_ps(b1, a3, c31);

        v8sf a4 = _mm256_broadcast_ss(A + 4 * (DIM * 2) + i);
        c40 = _mm256_fmadd_ps(b0, a4, c40);
        c41 = _mm256_fmadd_ps(b1, a4, c41);

        v8sf a5 = _mm256_broadcast_ss(A + 5 * (DIM * 2) + i);
        c50 = _mm256_fmadd_ps(b0, a5, c50);
        c51 = _mm256_fmadd_ps(b1, a5, c51);
    }

    c00 = _mm256_max_ps(c00, _mm256_setzero_ps());
    c01 = _mm256_max_ps(c01, _mm256_setzero_ps());
    c10 = _mm256_max_ps(c10, _mm256_setzero_ps());
    c11 = _mm256_max_ps(c11, _mm256_setzero_ps());
    c20 = _mm256_max_ps(c20, _mm256_setzero_ps());
    c21 = _mm256_max_ps(c21, _mm256_setzero_ps());
    c30 = _mm256_max_ps(c30, _mm256_setzero_ps());
    c31 = _mm256_max_ps(c31, _mm256_setzero_ps());
    c40 = _mm256_max_ps(c40, _mm256_setzero_ps());
    c41 = _mm256_max_ps(c41, _mm256_setzero_ps());
    c50 = _mm256_max_ps(c50, _mm256_setzero_ps());
    c51 = _mm256_max_ps(c51, _mm256_setzero_ps());

    c00 = _mm256_add_ps(c00, c01);
    c10 = _mm256_add_ps(c10, c11);
    c20 = _mm256_add_ps(c20, c21);
    c30 = _mm256_add_ps(c30, c31);
    c40 = _mm256_add_ps(c40, c41);
    c50 = _mm256_add_ps(c50, c51);

    c00 = _mm256_hadd_ps(c00, c10);
    c20 = _mm256_hadd_ps(c20, c30);
    c40 = _mm256_hadd_ps(c40, c50);

    c00 = _mm256_hadd_ps(c00, c20);
    c40 = _mm256_hadd_ps(c40, c40);

    C[0] = c00[0] + c00[4];
    C[1] = c00[1] + c00[5];
    C[2] = c00[2] + c00[6];
    C[3] = c00[3] + c00[7];
    C[4] = c40[0] + c40[4];
    C[5] = c40[1] + c40[5];
}

static inline void swap(float **a, float **b)
{
    float *t = *a;
    *a = *b;
    *b = t;
}

void cgnn_graph_sage_16(int params, const float *param,
                        uint32_t N, const uint32_t *V, const uint32_t *E, int md,
                        float *x, float *y, double *p)
{
    int layers = params / ((DIM * 2) * DIM + DIM);
    const float *W = param;
    const float *bias = param + (DIM * 2) * DIM;

    float *buffer = aligned_alloc(64, sizeof(float) * (DIM * 2) * RM);

    // Input layer
    for (int b = 0; b < N; b += RM)
    {
        for (int bi = 0; bi < RM && bi + b < N; bi++)
            message_kernel_input(b + bi, N, V, E, md, buffer + bi * (DIM * 2));
        dense_kernel(buffer, W, bias, y + b * DIM);
    }
    swap(&x, &y);
    W = bias + DIM;
    bias = W + (DIM * 2) * DIM;

    for (int i = 1; i < layers - 1; i++)
    {
        // Hidden layers
        for (uint32_t b = 0; b < N; b += RM)
        {
            for (uint32_t bi = 0; bi < RM && bi + b < N; bi++)
                message_kernel(b + bi, N, V, E, x, buffer + bi * (DIM * 2));
            dense_kernel(buffer, W, bias, y + b * DIM);
        }
        swap(&x, &y);
        W = bias + DIM;
        bias = W + (DIM * 2) * DIM;
    }
    // Output layers
    for (uint32_t b = 0; b < N; b += RM)
    {
        for (uint32_t bi = 0; bi < RM && bi + b < N; bi++)
            message_kernel(b + bi, N, V, E, x, buffer + bi * (DIM * 2));
        dense_kernel_output(buffer, W, bias, p + b);
    }

    free(buffer);
}

void cgnn_graph_sage_16_par(int params, const float *param,
                            uint32_t N, const uint32_t *V, const uint32_t *E, int md,
                            float *x, float *y, double *p)
{

#pragma omp parallel firstprivate(x, y)
    {
        int layers = params / ((DIM * 2) * DIM + DIM);
        const float *W = param;
        const float *bias = param + (DIM * 2) * DIM;

        float *buffer = aligned_alloc(64, sizeof(float) * (DIM * 2) * RM);

        // Input layer
#pragma omp for schedule(dynamic, 256)
        for (uint32_t b = 0; b < N; b += RM)
        {
            for (uint32_t bi = 0; bi < RM && bi + b < N; bi++)
                message_kernel_input(b + bi, N, V, E, md, buffer + bi * (DIM * 2));
            dense_kernel(buffer, W, bias, y + b * DIM);
        }
        swap(&x, &y);
        W = bias + DIM;
        bias = W + (DIM * 2) * DIM;

        for (int i = 1; i < layers - 1; i++)
        {
            // Hidden layers
#pragma omp for schedule(dynamic, 256)
            for (uint32_t b = 0; b < N; b += RM)
            {
                for (uint32_t bi = 0; bi < RM && bi + b < N; bi++)
                    message_kernel(b + bi, N, V, E, x, buffer + bi * (DIM * 2));
                dense_kernel(buffer, W, bias, y + b * DIM);
            }
            swap(&x, &y);
            W = bias + DIM;
            bias = W + (DIM * 2) * DIM;
        }
        // Output layers
#pragma omp for schedule(dynamic, 256)
        for (uint32_t b = 0; b < N; b += RM)
        {
            for (uint32_t bi = 0; bi < RM && bi + b < N; bi++)
                message_kernel(b + bi, N, V, E, x, buffer + bi * (DIM * 2));
            dense_kernel_output(buffer, W, bias, p + b);
        }

        free(buffer);
    }
}
