#include "smallest_degree_last.h"
#include "coloring.h"
#include "runner.h"

#include <stdlib.h>
#include <stdarg.h>
#include <omp.h>

struct sl_data
{
    uint32_t *degree;
    uint32_t *next, *prev, *remove;
    uint32_t **next_buff, **remove_buff;
};

void *smallest_degree_last_setup(uint32_t N, uint32_t M)
{
    struct sl_data *d = malloc(sizeof(struct sl_data));

    d->degree = malloc(sizeof(uint32_t) * N);
    d->next = malloc(sizeof(uint32_t) * N);
    d->prev = malloc(sizeof(uint32_t) * N);
    d->remove = malloc(sizeof(uint32_t) * N);

#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();
#pragma omp single
        {
            d->next_buff = malloc(sizeof(uint32_t *) * nt);
            d->remove_buff = malloc(sizeof(uint32_t *) * nt);
        }
        d->next_buff[tid] = malloc(sizeof(uint32_t) * N);
        d->remove_buff[tid] = malloc(sizeof(uint32_t) * N);
    }

    return (void *)d;
}

void smallest_degree_last_cleanup(void *sl)
{
    struct sl_data *d = (struct sl_data *)sl;

    free(d->degree);
    free(d->next);
    free(d->prev);
    free(d->remove);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        free(d->next_buff[tid]);
        free(d->remove_buff[tid]);
    }

    free(d->next_buff);
    free(d->remove_buff);

    free(d);
}

void smallest_degree_last_ordering_internal(uint32_t N, const uint32_t *V, const uint32_t *E, void *sl, double *p)
{
    struct sl_data *d = (struct sl_data *)sl;

    uint32_t min_degree = N;
    for (uint32_t u = 0; u < N; u++)
    {
        uint32_t deg = V[u + 1] - V[u];
        d->degree[u] = deg;
        d->prev[u] = u;
        p[u] = -1.0f;
        if (deg < min_degree)
            min_degree = deg;
    }

    uint32_t delta = min_degree;
    uint32_t prev = N, next = 0, remove = 0;
    while (prev > 0)
    {
        min_degree = N;
        for (uint32_t i = 0; i < prev; i++)
        {
            uint32_t u = d->prev[i];

            if (d->degree[u] <= delta)
            {
                d->remove[remove++] = u;
                p[u] = delta;
            }
            else
            {
                d->next[next++] = u;
                if (d->degree[u] < min_degree)
                    min_degree = d->degree[u];
            }
        }
        for (uint32_t i = 0; i < remove; i++)
        {
            uint32_t u = d->remove[i];
            for (uint32_t j = V[u]; j < V[u + 1]; j++)
            {
                uint32_t v = E[j];
                if (p[v] >= 0.0f)
                    continue;

                d->degree[v]--;
                if (d->degree[v] < min_degree)
                    min_degree = d->degree[v];
            }
        }
        delta++;
        if (min_degree > delta)
            delta = min_degree;

        prev = next;
        next = 0;
        remove = 0;

        uint32_t *t = d->prev;
        d->prev = d->next;
        d->next = t;
    }
}

void smallest_degree_last_ordering(uint32_t N, const uint32_t *V, const uint32_t *E, double *p)
{
    void *sl = smallest_degree_last_setup(N, V[N]);
    smallest_degree_last_ordering_internal(N, V, E, sl, p);
    smallest_degree_last_cleanup(sl);
}

void smallest_degree_last_internal(uint32_t N, const uint32_t *V, const uint32_t *E,
                                   void *sl, void *gd, double *p, int *color)
{
    smallest_degree_last_ordering_internal(N, V, E, sl, p);
    greedy_internal(N, V, E, gd, p, color);
}

int *smallest_degree_last_coloring(uint32_t N, const uint32_t *V, const uint32_t *E)
{
    int *color = malloc(sizeof(int) * N);
    void *sl = smallest_degree_last_setup(N, V[N]);
    void *gd = greedy_setup(N, V[N]);
    double *p = malloc(sizeof(double) * N);

    smallest_degree_last_internal(N, V, E, sl, gd, p, color);

    smallest_degree_last_cleanup(sl);
    greedy_cleanup(gd);
    free(p);
    return color;
}

void smallest_degree_last_internal_par(uint32_t N, const uint32_t *V, const uint32_t *E,
                                       void *sl, void *jp, double *p, int *color)
{
    struct sl_data *d = (struct sl_data *)sl;

    uint32_t min_degree = N;
#pragma omp parallel for reduction(min : min_degree)
    for (uint32_t u = 0; u < N; u++)
    {
        uint32_t deg = V[u + 1] - V[u];
        d->degree[u] = deg;
        d->prev[u] = u;
        p[u] = -1.0f;
        if (deg < min_degree)
            min_degree = deg;
    }

    uint32_t delta = min_degree;
    uint32_t prev = N, next = 0, remove = 0;
    min_degree = N;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        while (prev > 0)
        {
            uint32_t n_next = 0, n_remove = 0;
#pragma omp for nowait reduction(min : min_degree)
            for (uint32_t i = 0; i < prev; i++)
            {
                uint32_t u = d->prev[i];

                if (d->degree[u] <= delta)
                {
                    d->remove_buff[tid][n_remove++] = u;
                    p[u] = delta;
                }
                else
                {
                    d->next_buff[tid][n_next++] = u;
                    if (d->degree[u] < min_degree)
                        min_degree = d->degree[u];
                }
            }

            uint32_t pos = __atomic_fetch_add(&remove, n_remove, __ATOMIC_RELAXED);
            for (uint32_t i = 0; i < n_remove; i++)
                d->remove[pos + i] = d->remove_buff[tid][i];

            pos = __atomic_fetch_add(&next, n_next, __ATOMIC_RELAXED);
            for (uint32_t i = 0; i < n_next; i++)
                d->next[pos + i] = d->next_buff[tid][i];

#pragma omp barrier

#pragma omp for reduction(min : min_degree)
            for (uint32_t i = 0; i < remove; i++)
            {
                uint32_t u = d->remove[i];
                for (uint32_t j = V[u]; j < V[u + 1]; j++)
                {
                    uint32_t v = E[j];
                    if (p[v] >= 0.0f)
                        continue;

                    uint32_t deg = __atomic_sub_fetch(d->degree + v, 1, __ATOMIC_RELAXED);
                    if (deg < min_degree)
                        min_degree = deg;
                }
            }

#pragma omp single
            {
                delta++;
                if (min_degree > delta)
                    delta = min_degree;

                min_degree = N;

                prev = next;
                next = 0;
                remove = 0;

                uint32_t *t = d->prev;
                d->prev = d->next;
                d->next = t;
            }
        }
    }

    jones_plassmann_internal(N, V, E, jp, p, color);
}

int *smallest_degree_last_coloring_par(uint32_t N, const uint32_t *V, const uint32_t *E)
{
    int *color = malloc(sizeof(int) * N);
    void *sl = smallest_degree_last_setup(N, V[N]);
    void *jp = jones_plassmann_setup(N, V[N]);
    double *p = malloc(sizeof(double) * N);

    smallest_degree_last_internal_par(N, V, E, sl, jp, p, color);

    smallest_degree_last_cleanup(sl);
    jones_plassmann_cleanup(jp);
    free(p);
    return color;
}

void smallest_degree_last_runner(uint32_t N, const uint32_t *V, const uint32_t *E,
                                 int it, int k, int t_tot, int argc, ...)
{
    va_list argv;
    va_start(argv, argc);
    runner_test_full(N, V, E,
                     smallest_degree_last_setup,
                     smallest_degree_last_cleanup,
                     smallest_degree_last_internal,
                     smallest_degree_last_internal_par,
                     "SL", it, k, t_tot, argc, argv);
    va_end(argv);
}

void smallest_log_degree_last_ordering_internal(uint32_t N, const uint32_t *V, const uint32_t *E, void *sl, double *p)
{
    struct sl_data *d = (struct sl_data *)sl;

    uint32_t min_degree = N;
    for (uint32_t u = 0; u < N; u++)
    {
        uint32_t deg = V[u + 1] - V[u];
        d->degree[u] = deg;
        d->prev[u] = u;
        if (deg < min_degree)
            min_degree = deg;
    }

    uint32_t delta = 0;
    while ((1 << delta) < min_degree)
        delta++;
    uint32_t prev = N, next = 0, remove = 0;

    while (prev > 0)
    {
        for (uint32_t i = 0; i < prev; i++)
        {
            uint32_t u = d->prev[i];

            if (d->degree[u] <= (1 << delta))
            {
                d->remove[remove++] = u;
                p[u] = delta;
            }
            else
            {
                d->next[next++] = u;
            }
        }
        for (uint32_t i = 0; i < remove; i++)
        {
            uint32_t u = d->remove[i];
            for (uint32_t j = V[u]; j < V[u + 1]; j++)
            {
                uint32_t v = E[j];
                d->degree[v]--;
            }
        }
        delta++;

        prev = next;
        next = 0;
        remove = 0;

        uint32_t *t = d->prev;
        d->prev = d->next;
        d->next = t;
    }
}

void smallest_log_degree_last_ordering(uint32_t N, const uint32_t *V, const uint32_t *E, double *p)
{
    void *sl = smallest_degree_last_setup(N, V[N]);
    smallest_log_degree_last_ordering_internal(N, V, E, sl, p);
    smallest_degree_last_cleanup(sl);
}

void smallest_log_degree_last_internal(uint32_t N, const uint32_t *V, const uint32_t *E,
                                       void *sl, void *gd, double *p, int *color)
{
    smallest_log_degree_last_ordering_internal(N, V, E, sl, p);
    greedy_internal(N, V, E, gd, p, color);
}

int *smallest_log_degree_last_coloring(uint32_t N, const uint32_t *V, const uint32_t *E)
{
    int *colors = malloc(sizeof(int) * N);
    void *sl = smallest_degree_last_setup(N, V[N]);
    void *gd = greedy_setup(N, V[N]);
    double *p = malloc(sizeof(double) * N);

    smallest_log_degree_last_internal(N, V, E, sl, gd, p, colors);

    smallest_degree_last_cleanup(sl);
    greedy_cleanup(gd);
    free(p);
    return colors;
}

void smallest_log_degree_last_internal_par(uint32_t N, const uint32_t *V, const uint32_t *E,
                                           void *sl, void *jp, double *p, int *color)
{
    struct sl_data *d = (struct sl_data *)sl;

    uint32_t min_degree = N;
#pragma omp parallel for reduction(min : min_degree)
    for (uint32_t u = 0; u < N; u++)
    {
        uint32_t deg = V[u + 1] - V[u];
        d->degree[u] = deg;
        d->prev[u] = u;
        p[u] = -1.0f;
        if (deg < min_degree)
            min_degree = deg;
    }

    uint32_t delta = 0;
    while ((1 << delta) < min_degree)
        delta++;
    uint32_t prev = N, next = 0, remove = 0;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        while (prev > 0)
        {
            uint32_t n_next = 0, n_remove = 0;
#pragma omp for nowait
            for (uint32_t i = 0; i < prev; i++)
            {
                uint32_t u = d->prev[i];

                if (d->degree[u] <= (1 << delta))
                {
                    d->remove_buff[tid][n_remove++] = u;
                    p[u] = delta;
                }
                else
                {
                    d->next_buff[tid][n_next++] = u;
                }
            }

            uint32_t pos = __atomic_fetch_add(&remove, n_remove, __ATOMIC_RELAXED);
            for (uint32_t i = 0; i < n_remove; i++)
                d->remove[pos + i] = d->remove_buff[tid][i];

            pos = __atomic_fetch_add(&next, n_next, __ATOMIC_RELAXED);
            for (uint32_t i = 0; i < n_next; i++)
                d->next[pos + i] = d->next_buff[tid][i];

#pragma omp barrier

#pragma omp for
            for (uint32_t i = 0; i < remove; i++)
            {
                uint32_t u = d->remove[i];
                for (uint32_t j = V[u]; j < V[u + 1]; j++)
                {
                    uint32_t v = E[j];
                    if (p[v] >= 0.0f)
                        continue;

                    uint32_t deg = __atomic_sub_fetch(d->degree + v, 1, __ATOMIC_RELAXED);
                }
            }

#pragma omp single
            {
                delta++;

                prev = next;
                next = 0;
                remove = 0;

                uint32_t *t = d->prev;
                d->prev = d->next;
                d->next = t;
            }
        }
    }

    jones_plassmann_internal(N, V, E, jp, p, color);
}

int *smallest_log_degree_last_coloring_par(uint32_t N, const uint32_t *V, const uint32_t *E)
{
    int *colors = malloc(sizeof(int) * N);
    void *sl = smallest_degree_last_setup(N, V[N]);
    void *jp = jones_plassmann_setup(N, V[N]);
    double *p = malloc(sizeof(double) * N);

    smallest_log_degree_last_internal_par(N, V, E, sl, jp, p, colors);

    smallest_degree_last_cleanup(sl);
    jones_plassmann_cleanup(jp);
    free(p);
    return colors;
}

void smallest_log_degree_last_runner(uint32_t N, const uint32_t *V, const uint32_t *E,
                                     int it, int k, int t_tot, int argc, ...)
{
    va_list argv;
    va_start(argv, argc);
    runner_test_full(N, V, E,
                     smallest_degree_last_setup,
                     smallest_degree_last_cleanup,
                     smallest_log_degree_last_internal,
                     smallest_log_degree_last_internal_par,
                     "SLL", it, k, t_tot, argc, argv);
    va_end(argv);
}
