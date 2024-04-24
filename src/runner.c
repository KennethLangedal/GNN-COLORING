#include "runner.h"
#include "coloring.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void runner_test_ordering(uint32_t N, const uint32_t *V, const uint32_t *E, const char *name, int k,
                          void (*f)(uint32_t, const uint32_t *, const uint32_t *, double *))
{
    double *p = malloc(sizeof(double) * N);
    void *jp = jones_plassmann_setup(N, V[N]);
    int *color = malloc(sizeof(int) * N);

    printf("%s: ", name);
    fflush(stdout);

    f(N, V, E, p);
    jones_plassmann_internal(N, V, E, jp, p, color);
    int C = validate_coloring_par(N, V, E, color);

    printf("%d ", C);
    fflush(stdout);

    for (int i = 1; i < k; i++)
    {
#pragma omp parallel for
        for (uint32_t j = 0; j < N; j++)
            p[j] = color[j];

        jones_plassmann_internal(N, V, E, jp, p, color);
        C = validate_coloring_par(N, V, E, color);

        printf("%d ", C);
        fflush(stdout);
    }

    printf("\n");

    free(p);
    jones_plassmann_cleanup(jp);
    free(color);
}

void runner_test_ordering_time(uint32_t N, const uint32_t *V, const uint32_t *E, const char *name,
                               void (*f)(uint32_t, const uint32_t *, const uint32_t *, double *))
{
    double *p = malloc(sizeof(double) * N);
    void *gd = greedy_setup(N, V[N]);
    int *color = malloc(sizeof(int) * N);

    printf("%s: ", name);
    fflush(stdout);

    double best = 99999.9;
    for (int i = 0; i < 1; i++)
    {
        double t0 = omp_get_wtime();
        f(N, V, E, p);
        greedy_internal(N, V, E, gd, p, color);
        double t1 = omp_get_wtime();
        if (t1 - t0 < best)
            best = t1 - t0;
    }
    int C = validate_coloring_par(N, V, E, color);

    printf("%d %.4lf\n", C, best);

    free(p);
    greedy_cleanup(gd);
    free(color);
}

void validate(uint32_t N, const uint32_t *V, const uint32_t *E, int *colors, int *C)
{
    int c = validate_coloring_par(N, V, E, colors);
    if (c == 0)
        printf("Invalid coloring\n");
    else if (*C < 0)
        *C = c;
    else if (*C != c)
        printf("Unexpected number of colors (%d %d)\n", *C, c);

    // printf("*");
    // fflush(stdout);
}

void runner_test_full(uint32_t N, const uint32_t *V, const uint32_t *E,
                      void *(*setup)(uint32_t, uint32_t),
                      void (*cleanup)(void *),
                      void (*f_seq)(uint32_t, const uint32_t *, const uint32_t *, void *, void *, double *, int *),
                      void (*f_par)(uint32_t, const uint32_t *, const uint32_t *, void *, void *, double *, int *),
                      const char *s, int it, int k, int t_tot, int argc, va_list argv)
{
    int nt[argc];
    for (int i = 0; i < argc; i++)
        nt[i] = va_arg(argv, int);

    int n_colors[k];
    double times[it][k][argc + 1];
    for (int i = 0; i < k; i++)
        n_colors[i] = -1;

    omp_set_num_threads(t_tot);

    int *colors = malloc(sizeof(int) * N);
    double *p = aligned_alloc(32, sizeof(double) * (N + 8));
    void *fd = setup(N, V[N]);
    void *jp = jones_plassmann_setup(N, V[N]);
    void *gd = greedy_setup(N, V[N]);

    for (int t = 0; t < it; t++)
    {
        // Sequential run
        double t0 = omp_get_wtime();
        f_seq(N, V, E, fd, gd, p, colors);
        double t1 = omp_get_wtime();

        times[t][0][0] = t1 - t0;
        validate(N, V, E, colors, &n_colors[0]);

        for (int i = 1; i < k; i++)
        {
            t0 = omp_get_wtime();
            for (uint32_t u = 0; u < N; u++)
                p[u] = colors[u];
            greedy_internal(N, V, E, gd, p, colors);
            t1 = omp_get_wtime();

            times[t][i][0] = (t1 - t0) + times[t][i - 1][0];
            validate(N, V, E, colors, &n_colors[i]);
        }

        // Parallel run
        for (int i = 0; i < argc; i++)
        {
            omp_set_num_threads(nt[i]);
            t0 = omp_get_wtime();
            f_par(N, V, E, fd, jp, p, colors);
            t1 = omp_get_wtime();
            times[t][0][i + 1] = t1 - t0;

            omp_set_num_threads(t_tot);
            validate(N, V, E, colors, &n_colors[0]);

            for (int j = 1; j < k; j++)
            {
                omp_set_num_threads(nt[i]);
                t0 = omp_get_wtime();
#pragma omp parallel for
                for (uint32_t u = 0; u < N; u++)
                    p[u] = colors[u];
                jones_plassmann_internal(N, V, E, jp, p, colors);
                t1 = omp_get_wtime();

                times[t][j][i + 1] = (t1 - t0) + times[t][j - 1][i + 1];
                omp_set_num_threads(t_tot);
                validate(N, V, E, colors, &n_colors[j]);
            }
        }
    }

    // printf("\n");
    for (int i = 0; i < k; i++)
    {
        printf("%s%d %d ", s, i, n_colors[i]);
        for (int j = 0; j < argc + 1; j++)
        {
            double best = 9999.9;
            for (int t = 0; t < it; t++)
                if (times[t][i][j] < best)
                    best = times[t][i][j];

            printf("%.6lf ", best);
        }
        printf("\n");
    }

    free(colors);
    free(p);
    cleanup(fd);
    jones_plassmann_cleanup(jp);
    greedy_cleanup(gd);
}