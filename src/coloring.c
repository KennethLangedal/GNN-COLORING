#include "coloring.h"

#include <omp.h>
#include <stdint.h>
#include <stdlib.h>

int *greedy_coloring(uint32_t N, const uint32_t *V, const uint32_t *E, const double *p)
{
    int *color = malloc(sizeof(int) * N);
    void *gd = greedy_setup(N, V[N]);

    greedy_internal(N, V, E, gd, p, color);

    greedy_cleanup(gd);
    return color;
}

struct gd_data
{
    uint32_t *next, *prev, *count;
    uint32_t *pred, *E_split;

    uint32_t *marks;
};

void *greedy_setup(uint32_t N, uint32_t M)
{
    struct gd_data *d = malloc(sizeof(struct gd_data));

    // Shared arrays
    d->next = malloc(sizeof(uint32_t) * N);
    d->prev = malloc(sizeof(uint32_t) * N);
    d->count = malloc(sizeof(uint32_t) * N);

    // Split datastructure
    d->pred = malloc(sizeof(uint32_t) * N);
    d->E_split = malloc(sizeof(uint32_t) * M);

    d->marks = malloc(sizeof(uint32_t) * N);

    return (void *)d;
}

void greedy_cleanup(void *gd)
{
    struct gd_data *d = (struct gd_data *)gd;

    free(d->next);
    free(d->prev);
    free(d->count);

    free(d->pred);
    free(d->E_split);

    free(d->marks);

    free(d);
}

void greedy_internal(uint32_t N, const uint32_t *V, const uint32_t *E, void *gd, const double *p, int *color)
{
    uint32_t n_next = 0, n_prev = 0, max_degree = 0;
    struct gd_data *d = (struct gd_data *)gd;

    for (uint32_t u = 0; u < N; u++)
    {
        uint32_t j = V[u], k = V[u + 1] - 1;

        for (uint32_t i = V[u]; i < V[u + 1]; i++)
        {
            uint32_t v = E[i];
            if (p[v] > p[u] || (p[u] == p[v] && u > v))
                d->E_split[j++] = v;
            else
                d->E_split[k--] = v;
        }

        color[u] = 0;
        d->count[u] = j - V[u];
        d->pred[u] = j;

        if (d->count[u] == 0)
            d->prev[n_prev++] = u;

        if (V[u + 1] - V[u] > max_degree)
            max_degree = V[u + 1] - V[u];
    }

    for (uint32_t i = 0; i < max_degree + 1; i++)
        d->marks[i] = N;

    while (n_prev > 0)
    {
        for (uint32_t i = 0; i < n_prev; i++)
        {
            uint32_t u = d->prev[i];

            // Find lowest available color
            for (uint32_t j = V[u]; j < d->pred[u]; j++)
                d->marks[color[d->E_split[j]]] = u;

            int c = 1;
            while (d->marks[c] == u)
                c++;
            color[u] = c;

            // Updete uncolored neighbors
            for (uint32_t j = d->pred[u]; j < V[u + 1]; j++)
            {
                uint32_t v = d->E_split[j];

                d->count[v]--;
                if (d->count[v] == 0)
                    d->next[n_next++] = v;
            }
        }

        n_prev = n_next;
        n_next = 0;

        uint32_t *t = d->next;
        d->next = d->prev;
        d->prev = t;
    }
}

int validate_coloring(uint32_t N, const uint32_t *V, const uint32_t *E, const int *color)
{
    int f = 0, C = 1;
    for (uint32_t u = 0; u < N; u++)
    {
        if (color[u] <= 0)
            f = 1;
        for (uint32_t i = V[u]; i < V[u + 1]; i++)
        {
            uint32_t v = E[i];
            if (color[u] == color[v])
                f = 1;
        }
        if (C < color[u])
            C = color[u];
    }
    if (f)
        return 0;
    return C;
}

int *jones_plassmann_coloring(uint32_t N, const uint32_t *V, const uint32_t *E, const double *p)
{
    int *color = malloc(sizeof(int) * N);
    void *jp = jones_plassmann_setup(N, V[N]);

    jones_plassmann_internal(N, V, E, jp, p, color);

    jones_plassmann_cleanup(jp);
    return color;
}

struct jp_data
{
    uint32_t *next, *prev, *count;
    uint32_t *pred, *E_split;

    uint32_t **buff, **marks;
    uint32_t *max_degree;
};

void *jones_plassmann_setup(uint32_t N, uint32_t M)
{
    struct jp_data *d = malloc(sizeof(struct jp_data));

    // Shared arrays
    d->next = malloc(sizeof(uint32_t) * N);
    d->prev = malloc(sizeof(uint32_t) * N);
    d->count = malloc(sizeof(uint32_t) * N);
    d->pred = malloc(sizeof(uint32_t) * N);
    d->E_split = malloc(sizeof(uint32_t) * M);

#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();

        // Local arrays
#pragma omp single
        {
            d->buff = malloc(sizeof(uint32_t *) * nt);
            d->marks = malloc(sizeof(uint32_t *) * nt);
            d->max_degree = malloc(sizeof(uint32_t) * nt);
        }

        d->buff[tid] = malloc(sizeof(uint32_t) * M);
        d->marks[tid] = malloc(sizeof(uint32_t) * N);
    }

    return (void *)d;
}

void jones_plassmann_cleanup(void *jp)
{
    struct jp_data *d = (struct jp_data *)jp;

    free(d->next);
    free(d->prev);
    free(d->count);
    free(d->pred);
    free(d->E_split);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        free(d->buff[tid]);
        free(d->marks[tid]);
    }

    free(d->buff);
    free(d->marks);
    free(d->max_degree);

    free(d);
}

void jones_plassmann_internal(uint32_t N, const uint32_t *V, const uint32_t *E,
                              void *jp, const double *p, int *color)
{
    uint32_t n_next = 0, n_prev = 0;
    struct jp_data *d = (struct jp_data *)jp;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();

        uint32_t n_local = 0;
        uint32_t *buff = d->buff[tid];
        uint32_t *marks = d->marks[tid];

        uint32_t max_degree = 0;

        // Split E
#pragma omp for nowait
        for (uint32_t u = 0; u < N; u++)
        {
            uint32_t j = V[u], k = V[u + 1] - 1;

            for (uint32_t i = V[u]; i < V[u + 1]; i++)
            {
                uint32_t v = E[i];
                if (p[v] > p[u] || (p[u] == p[v] && u > v))
                    d->E_split[j++] = v;
                else
                    d->E_split[k--] = v;
            }

            color[u] = 0;
            d->count[u] = j - V[u];
            d->pred[u] = j;

            if (d->count[u] == 0)
                buff[n_local++] = u;

            if (V[u + 1] - V[u] > max_degree)
                max_degree = V[u + 1] - V[u];
        }

        // Copy to shared buffer
        uint32_t pos = __atomic_fetch_add(&n_prev, n_local, __ATOMIC_RELAXED);
        for (uint32_t i = 0; i < n_local; i++)
            d->prev[pos + i] = buff[i];

        d->max_degree[tid] = max_degree;

#pragma omp barrier

        for (uint32_t i = 0; i < nt; i++)
            if (d->max_degree[i] > max_degree)
                max_degree = d->max_degree[i];

        for (uint32_t i = 0; i < max_degree + 1; i++)
            marks[i] = -1;

        // While frontier is not empty
        while (n_prev > 0)
        {
            n_local = 0;

#pragma omp for nowait
            for (uint32_t i = 0; i < n_prev; i++)
            {
                uint32_t u = d->prev[i];

                // Find lowest available color
                for (uint32_t j = V[u]; j < d->pred[u]; j++)
                    marks[color[d->E_split[j]]] = u;

                int c = 1;
                while (marks[c] == u)
                    c++;
                color[u] = c;

                // Updete uncolored neighbors
                for (uint32_t j = d->pred[u]; j < V[u + 1]; j++)
                {
                    uint32_t v = d->E_split[j];

                    uint32_t vc = __atomic_sub_fetch(d->count + v, 1, __ATOMIC_RELAXED);
                    if (vc == 0)
                        buff[n_local++] = v;
                }
            }

            // Copy local findings to shared frontier
            pos = __atomic_fetch_add(&n_next, n_local, __ATOMIC_RELAXED);
            for (uint32_t i = 0; i < n_local; i++)
                d->next[pos + i] = buff[i];

#pragma omp barrier

#pragma omp single
            {
                n_prev = n_next;
                n_next = 0;

                uint32_t *t = d->next;
                d->next = d->prev;
                d->prev = t;
            }
        }
    }
}

int validate_coloring_par(uint32_t N, const uint32_t *V, const uint32_t *E, const int *color)
{
    int f = 0, C = 1;
#pragma omp parallel for reduction(max : f, C)
    for (uint32_t u = 0; u < N; u++)
    {
        if (color[u] <= 0)
            f = 1;
        for (uint32_t i = V[u]; i < V[u + 1]; i++)
        {
            uint32_t v = E[i];
            if (color[u] == color[v])
                f = 1;
        }
        if (C < color[u])
            C = color[u];
    }
    if (f)
        return 0;
    return C;
}