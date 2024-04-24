#include "fast_io.h"

#include "coloring.h"
#include "runner.h"
#include "first_fit.h"
#include "largest_degree_first.h"
#include "smallest_degree_last.h"
#include "incidence_degree.h"
#include "gnn.h"

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

typedef struct
{
    int ff, lf, llf, sl, sll, id, sda, sd, gnn2, gnn3, gnn4;
} config;

void test_all(uint32_t N, uint32_t *V, uint32_t *E, config c)
{
    void *jp = jones_plassmann_setup(N, V[N]);
    double *p = malloc(sizeof(double) * (N + 8));
    int *color = malloc(sizeof(int) * N);

    if (c.ff)
    {
        first_fit_ordering(N, V, E, p);
        jones_plassmann_internal(N, V, E, jp, p, color);
        int ff = validate_coloring_par(N, V, E, color);
        printf("%d ", ff);
    }

    if (c.lf)
    {
        largest_degree_first_ordering(N, V, E, p);
        jones_plassmann_internal(N, V, E, jp, p, color);
        int lf = validate_coloring_par(N, V, E, color);
        printf("%d ", lf);
    }

    if (c.llf)
    {
        largest_log_degree_first_ordering(N, V, E, p);
        jones_plassmann_internal(N, V, E, jp, p, color);
        int llf = validate_coloring_par(N, V, E, color);
        printf("%d ", llf);
    }

    if (c.sl)
    {
        smallest_degree_last_ordering(N, V, E, p);
        jones_plassmann_internal(N, V, E, jp, p, color);
        int sl = validate_coloring_par(N, V, E, color);
        printf("%d ", sl);
    }

    if (c.sll)
    {
        smallest_log_degree_last_ordering(N, V, E, p);
        jones_plassmann_internal(N, V, E, jp, p, color);
        int sll = validate_coloring_par(N, V, E, color);
        printf("%d ", sll);
    }

    if (c.id)
    {
        incidence_degree_ordering(N, V, E, p);
        jones_plassmann_internal(N, V, E, jp, p, color);
        int id = validate_coloring_par(N, V, E, color);
        printf("%d ", id);
    }

    if (c.sda)
    {
        saturation_degree_ordering_alt(N, V, E, p);
        jones_plassmann_internal(N, V, E, jp, p, color);
        int sda = validate_coloring_par(N, V, E, color);
        printf("%d ", sda);
    }

    if (c.sd)
    {
        saturation_degree_ordering(N, V, E, p);
        jones_plassmann_internal(N, V, E, jp, p, color);
        int sd = validate_coloring_par(N, V, E, color);
        printf("%d ", sd);
    }

    if (c.gnn2)
    {
        gnn_ordering(N, V, E, p, 2);
        jones_plassmann_internal(N, V, E, jp, p, color);
        int gnn2 = validate_coloring_par(N, V, E, color);
        printf("%d ", gnn2);
    }

    if (c.gnn3)
    {
        gnn_ordering(N, V, E, p, 3);
        jones_plassmann_internal(N, V, E, jp, p, color);
        int gnn3 = validate_coloring_par(N, V, E, color);
        printf("%d ", gnn3);
    }

    if (c.gnn4)
    {
        gnn_ordering(N, V, E, p, 4);
        jones_plassmann_internal(N, V, E, jp, p, color);
        int gnn4 = validate_coloring_par(N, V, E, color);
        printf("%d ", gnn4);
    }

    printf("\n");

    jones_plassmann_cleanup(jp);
    free(p);
    free(color);
}

int main(int argc, char **argv)
{
    double t0 = omp_get_wtime();
    uint32_t N, *V, *E;
    FILE *f = fopen(argv[1], "r");
    mtx_parse_par(f, &N, &V, &E);
    fclose(f);
    double t1 = omp_get_wtime(), t2;

    // printf("Parsing took: %.4lfs, |V|=%u, |E|=%u\n", t1 - t0, N, V[N]);

    t0 = omp_get_wtime();
    graph_sort_edges_par(N, V, E);
    t1 = omp_get_wtime();

    // printf("Sorting edges took: %.4lfs\n", t1 - t0);

    t0 = omp_get_wtime();
    graph_make_simple(N, V, &E);
    t1 = omp_get_wtime();

    // printf("Making the graph simple took %.4lfs, new size |V|=%u, |E|=%u\n", t1 - t0, N, V[N]);

    // printf("%s %u %u\n", argv[1], N, V[N]);

    // first_fit_runner(N, V, E, 5, 1, 32, 6, 1, 2, 4, 8, 16, 32);
    // largest_degree_first_runner(N, V, E, 5, 1, 32, 6, 1, 2, 4, 8, 16, 32);
    // largest_log_degree_first_runner(N, V, E, 5, 1, 32, 6, 1, 2, 4, 8, 16, 32);
    // smallest_degree_last_runner(N, V, E, 5, 1, 32, 6, 1, 2, 4, 8, 16, 32);
    // smallest_log_degree_last_runner(N, V, E, 5, 1, 32, 6, 1, 2, 4, 8, 16, 32);
    // runner_test_ordering_time(N, V, E, "ID", incidence_degree_ordering);
    // runner_test_ordering_time(N, V, E, "SD", saturation_degree_ordering);

    printf("%s %d %d ", argv[1], N, V[N]);

    config c = {.gnn2 = 1, .gnn3 = 1, .gnn4 = 1};
    test_all(N, V, E, c);

    // runner_test_ordering_time(N, V, E, "FF", first_fit_ordering);
    // runner_test_ordering_time(N, V, E, "LF", largest_degree_first_ordering);
    // runner_test_ordering_time(N, V, E, "LLF", largest_log_degree_first_ordering);
    // runner_test_ordering_time(N, V, E, "SL", smallest_degree_last_ordering);
    // runner_test_ordering_time(N, V, E, "SLL", smallest_log_degree_last_ordering);
    // runner_test_ordering_time(N, V, E, "ID", incidence_degree_ordering);
    // runner_test_ordering_time(N, V, E, "SD", saturation_degree_ordering);
    // printf("\n");

    free(V);
    free(E);

    return 0;
}