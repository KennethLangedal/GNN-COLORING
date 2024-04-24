#include "largest_degree_first.h"
#include "coloring.h"
#include "runner.h"

#include <stdlib.h>
#include <omp.h>

/*
    Largest Degree First
*/

void largest_degree_first_ordering(uint32_t N, const uint32_t *V, const uint32_t *E, double *p)
{
    for (uint32_t u = 0; u < N; u++)
        p[u] = V[u + 1] - V[u];
}

void *largest_degree_first_setup(uint32_t N, uint32_t M)
{
    return NULL;
}

void largest_degree_first_cleanup(void *lf)
{
}

void largest_degree_first_internal(uint32_t N, const uint32_t *V, const uint32_t *E,
                                   void *lf, void *gd, double *p, int *color)
{
    for (uint32_t u = 0; u < N; u++)
        p[u] = V[u + 1] - V[u];

    greedy_internal(N, V, E, gd, p, color);
}

int *largest_degree_first_coloring(uint32_t N, const uint32_t *V, const uint32_t *E)
{
    int *color = malloc(sizeof(int) * N);
    void *gd = greedy_setup(N, V[N]);
    double *p = malloc(sizeof(double) * N);

    largest_degree_first_internal(N, V, E, NULL, gd, p, color);

    greedy_cleanup(gd);
    free(p);
    return color;
}

void largest_degree_first_internal_par(uint32_t N, const uint32_t *V, const uint32_t *E,
                                       void *lf, void *jp, double *p, int *color)
{
#pragma omp parallel for
    for (uint32_t u = 0; u < N; u++)
        p[u] = V[u + 1] - V[u];

    jones_plassmann_internal(N, V, E, jp, p, color);
}

int *largest_degree_first_coloring_par(uint32_t N, const uint32_t *V, const uint32_t *E)
{
    int *color = malloc(sizeof(int) * N);
    void *jp = jones_plassmann_setup(N, V[N]);
    double *p = malloc(sizeof(double) * N);

    largest_degree_first_internal_par(N, V, E, NULL, jp, p, color);

    jones_plassmann_cleanup(jp);
    free(p);
    return color;
}

void largest_degree_first_runner(uint32_t N, const uint32_t *V, const uint32_t *E,
                                 int it, int k, int t_tot, int ntc, const int *nt)
{
    runner_test_full(N, V, E,
                     largest_degree_first_setup,
                     largest_degree_first_cleanup,
                     largest_degree_first_internal,
                     largest_degree_first_internal_par,
                     "LF", it, k, t_tot, ntc, nt);
}

/*
    Largest Log Degree First
*/

void largest_log_degree_first_ordering(uint32_t N, const uint32_t *V, const uint32_t *E, double *p)
{
    for (uint32_t u = 0; u < N; u++)
    {
        uint32_t deg = V[u + 1] - V[u];
        if (deg > 0)
            p[u] = 32 - __builtin_clz(deg);
        else
            p[u] = 0.0;
    }
}

void largest_log_degree_first_internal(uint32_t N, const uint32_t *V, const uint32_t *E,
                                       void *lf, void *gd, double *p, int *color)
{
    largest_log_degree_first_ordering(N, V, E, p);
    greedy_internal(N, V, E, gd, p, color);
}

int *largest_log_degree_first_coloring(uint32_t N, const uint32_t *V, const uint32_t *E)
{
    int *color = malloc(sizeof(int) * N);
    void *gd = greedy_setup(N, V[N]);
    double *p = malloc(sizeof(double) * N);

    largest_log_degree_first_internal(N, V, E, NULL, gd, p, color);

    greedy_cleanup(gd);
    free(p);
    return color;
}

void largest_log_degree_first_ordering_par(uint32_t N, const uint32_t *V, const uint32_t *E, double *p)
{
#pragma omp parallel for
    for (uint32_t u = 0; u < N; u++)
    {
        uint32_t deg = V[u + 1] - V[u];
        if (deg > 0)
            p[u] = 32 - __builtin_clz(deg);
        else
            p[u] = 0.0;
    }
}

void largest_log_degree_first_internal_par(uint32_t N, const uint32_t *V, const uint32_t *E,
                                           void *lf, void *jp, double *p, int *color)
{

    largest_log_degree_first_ordering_par(N, V, E, p);
    jones_plassmann_internal(N, V, E, jp, p, color);
}

int *largest_log_degree_first_coloring_par(uint32_t N, const uint32_t *V, const uint32_t *E)
{
    int *color = malloc(sizeof(int) * N);
    void *jp = jones_plassmann_setup(N, V[N]);
    double *p = malloc(sizeof(double) * N);

    largest_log_degree_first_internal(N, V, E, NULL, jp, p, color);

    jones_plassmann_cleanup(jp);
    free(p);
    return color;
}

void largest_log_degree_first_runner(uint32_t N, const uint32_t *V, const uint32_t *E,
                                     int it, int k, int t_tot, int ntc, const int *nt)
{
    runner_test_full(N, V, E,
                     largest_degree_first_setup,
                     largest_degree_first_cleanup,
                     largest_log_degree_first_internal,
                     largest_log_degree_first_internal_par,
                     "LLF", it, k, t_tot, ntc, nt);
}