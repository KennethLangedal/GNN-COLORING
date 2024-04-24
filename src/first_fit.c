#include "first_fit.h"
#include "coloring.h"
#include "runner.h"

#include <stdlib.h>
#include <stdarg.h>
#include <omp.h>

void first_fit_ordering(uint32_t N, const uint32_t *V, const uint32_t *E, double *p)
{
    for (uint32_t i = 0; i < N; i++)
        p[i] = N - i;
}

void first_fit_ordering_par(uint32_t N, const uint32_t *V, const uint32_t *E, double *p)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < N; i++)
        p[i] = N - i;
}

int *first_fit_coloring(uint32_t N, const uint32_t *V, const uint32_t *E)
{
    double *p = malloc(sizeof(double) * N);
    first_fit_ordering(N, V, E, p);
    int *color = greedy_coloring(N, V, E, p);
    free(p);
    return color;
}

int *first_fit_coloring_par(uint32_t N, const uint32_t *V, const uint32_t *E)
{
    double *p = malloc(sizeof(double) * N);
    first_fit_ordering_par(N, V, E, p);
    int *color = jones_plassmann_coloring(N, V, E, p);
    free(p);
    return color;
}

void *first_fit_setup(uint32_t N, uint32_t M)
{
    int *marks = malloc(sizeof(int) * N);

    return (void *)marks;
}

void first_fit_cleanup(void *ff)
{
    int *marks = (int *)ff;

    free(marks);
}

void first_fit_internal(uint32_t N, const uint32_t *V, const uint32_t *E,
                        void *ff, void *gd, double *p, int *color)
{
    int *marks = (int *)ff;
    int max_degree = 0;

    for (int u = 0; u < N; u++)
    {
        color[u] = 0;
        if (V[u + 1] - V[u] > max_degree)
            max_degree = V[u + 1] - V[u];
    }

    for (int i = 0; i < max_degree + 1; i++)
        marks[i] = -1;

    for (uint32_t u = 0; u < N; u++)
    {
        for (uint32_t i = V[u]; i < V[u + 1]; i++)
            marks[color[E[i]]] = u;

        int c = 1;
        while (marks[c] == u)
            c++;

        color[u] = c;
    }
}

void first_fit_internal_par(uint32_t N, const uint32_t *V, const uint32_t *E,
                            void *ff, void *jp, double *p, int *color)
{
    first_fit_ordering_par(N, V, E, p);
    jones_plassmann_internal(N, V, E, jp, p, color);
}

void first_fit_runner(uint32_t N, const uint32_t *V, const uint32_t *E,
                      int it, int k, int t_tot, int argc, ...)
{
    va_list argv;
    va_start(argv, argc);
    runner_test_full(N, V, E,
                     first_fit_setup,
                     first_fit_cleanup,
                     first_fit_internal,
                     first_fit_internal_par,
                     "FF", it, k, t_tot, argc, argv);
    va_end(argv);
}