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
#include <unistd.h>
#include <ctype.h>

int test_ordering(uint32_t N, uint32_t *V, uint32_t *E,
                  void *jp, double *p, int *color, int K)
{
    jones_plassmann_internal(N, V, E, jp, p, color);
    for (int i = 1; i < K; i++)
    {
#pragma omp parallel for
        for (uint32_t j = 0; j < N; j++)
            p[j] = color[j];

        jones_plassmann_internal(N, V, E, jp, p, color);
    }
    return validate_coloring_par(N, V, E, color);
}

int main(int argc, char **argv)
{
    char *graph_path = "";
    int p = 0, s = 0, f = 0, k = 1, t = 1, v = 0;

    int c;
    opterr = 0;

    while ((c = getopt(argc, argv, "g:pcfk:t:vh")) != -1)
    {
        switch (c)
        {
        case 'g':
            graph_path = optarg;
            break;
        case 'p':
            p = 1;
            break;
        case 'c':
            s = 1;
            break;
        case 'f':
            f = 1;
            break;
        case 'k':
            k = atoi(optarg);
            break;
        case 't':
            t = atoi(optarg);
            break;
        case 'v':
            v = 1;
            break;
        case 'h':
            printf("Usage: ./coloring [options] tc1 tc2 ...\n");
            printf("tcx means number of threads to try\n");
            printf("Options are:\n");
            printf("-g path, sets input graph file\n");
            printf("-p, run parallel experiments\n");
            printf("-c, compute colors used fast\n");
            printf("-f, only do FF, LF, SL, and GNN\n");
            printf("-k #, number of repeated colorings\n");
            printf("-t #, total number of threads for faster I/O and validation\n");
            printf("-v, verbose\n");
            printf("-h, print this help message\n");
            break;

        default:
            printf("Unknown option %c\n", c);
            return 1;
        }
    }

    int ntc = argc - optind;
    int nt[ntc];
    for (int i = optind; i < argc; i++)
    {
        nt[i - optind] = atoi(argv[i]);
    }

    double t0 = omp_get_wtime();
    uint32_t N, *V, *E;
    FILE *fp = fopen(graph_path, "r");
    if (fp == NULL)
    {
        printf("Unable to open file %s\n", graph_path);
        return 1;
    }
    mtx_parse_par(fp, &N, &V, &E);
    fclose(fp);
    double t1 = omp_get_wtime(), t2;

    if (v)
        printf("Parsing took: %.4lfs, |V|=%u, |E|=%u\n", t1 - t0, N, V[N]);

    t0 = omp_get_wtime();
    graph_sort_edges_par(N, V, E);
    t1 = omp_get_wtime();

    if (v)
        printf("Sorting edges took: %.4lfs\n", t1 - t0);

    t0 = omp_get_wtime();
    graph_make_simple(N, V, &E);
    t1 = omp_get_wtime();

    if (v)
        printf("Making the graph simple took %.4lfs, new size |V|=%u, |E|=%u\n", t1 - t0, N, V[N]);

    if (s && !p)
    {
        void *jp = jones_plassmann_setup(N, V[N]);
        double *p = aligned_alloc(32, sizeof(double) * (N + 8));
        int *color = malloc(sizeof(int) * N);

        int ff = 0, lf = 0, llf = 0, sl = 0, sll = 0,
            id = 0, sda = 0, sd = 0, gnn2 = 0, gnn3 = 0, gnn4 = 0;

        first_fit_ordering_par(N, V, E, p);
        ff = test_ordering(N, V, E, jp, p, color, k);

        largest_degree_first_ordering(N, V, E, p);
        lf = test_ordering(N, V, E, jp, p, color, k);

        smallest_degree_last_ordering(N, V, E, p);
        sl = test_ordering(N, V, E, jp, p, color, k);

        gnn_ordering(N, V, E, p, 2);
        gnn2 = test_ordering(N, V, E, jp, p, color, k);

        gnn_ordering(N, V, E, p, 3);
        gnn3 = test_ordering(N, V, E, jp, p, color, k);

        gnn_ordering(N, V, E, p, 4);
        gnn4 = test_ordering(N, V, E, jp, p, color, k);

        if (!f)
        {
            largest_log_degree_first_ordering(N, V, E, p);
            llf = test_ordering(N, V, E, jp, p, color, k);

            smallest_log_degree_last_ordering(N, V, E, p);
            sll = test_ordering(N, V, E, jp, p, color, k);

            incidence_degree_ordering(N, V, E, p);
            id = test_ordering(N, V, E, jp, p, color, k);

            saturation_degree_ordering_alt(N, V, E, p);
            sda = test_ordering(N, V, E, jp, p, color, k);

            saturation_degree_ordering(N, V, E, p);
            sd = test_ordering(N, V, E, jp, p, color, k);
        }

        if (v)
            printf("ff lf llf sl sll id sda sd gnn2 gnn3 gnn4\n");
        printf("%d %d %d %d %d %d %d %d %d %d %d\n",
               ff, lf, llf, sl, sll, id, sda, sd, gnn2, gnn3, gnn4);

        jones_plassmann_cleanup(jp);
        free(p);
        free(color);
    }

    if (p)
    {
        if (v)
        {
            printf("name colors seq ");
            for (int i = 0; i < ntc; i++)
                printf("%d ", nt[i]);
            printf("\n");
        }
        first_fit_runner(N, V, E, 5, k, t, ntc, nt);
        largest_degree_first_runner(N, V, E, 5, k, t, ntc, nt);
        if (!f)
            largest_log_degree_first_runner(N, V, E, 5, k, t, ntc, nt);
        smallest_degree_last_runner(N, V, E, 5, k, t, ntc, nt);
        if (!f)
            smallest_log_degree_last_runner(N, V, E, 5, k, t, ntc, nt);
        gnn_runner(N, V, E, 2, 5, k, t, ntc, nt);
        gnn_runner(N, V, E, 3, 5, k, t, ntc, nt);
        gnn_runner(N, V, E, 4, 5, k, t, ntc, nt);
    }

    free(V);
    free(E);

    return 0;
}