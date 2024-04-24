#include "gnn.h"

#include "coloring.h"
#include "runner.h"

#include "fast_gnn.h"

#include <stdlib.h>
#include <stdio.h>

struct gnn_data
{
    int params, md;
    float *param;
    float *x, *y;
};

int layers = 2;

void *gnn_setup(uint32_t N, uint32_t M)
{
    struct gnn_data *d = malloc(sizeof(struct gnn_data));

    char path[256];
    sprintf(path, "models/m%d.gnn", layers);

    FILE *f = fopen(path, "r");
    if (f == NULL)
    {
        fprintf(stderr, "Failed to open model file (%d layers)\n", layers);
        exit(1);
    }
    int rc = fscanf(f, "%d %d", &d->params, &d->md);
    d->param = aligned_alloc(32, sizeof(float) * d->params);

    d->x = aligned_alloc(32, sizeof(float) * (N + 8) * 16);
    d->y = aligned_alloc(32, sizeof(float) * (N + 8) * 16);

    for (int i = 0; i < d->params; i++)
        rc = fscanf(f, "%f", d->param + i);
    fclose(f);

    return (void *)d;
}

void gnn_cleanup(void *ad)
{
    struct gnn_data *d = (struct gnn_data *)ad;

    free(d->param);
    free(d->x);
    free(d->y);

    free(d);
}

void gnn_ordering_internal(uint32_t N, const uint32_t *V, const uint32_t *E, void *ad, double *p)
{
    struct gnn_data *d = (struct gnn_data *)ad;

    cgnn_graph_sage_16(d->params, d->param, N, V, E, d->md, d->x, d->y, p);
}

void gnn_internal(uint32_t N, const uint32_t *V, const uint32_t *E,
                  void *ad, void *gd, double *p, int *color)
{
    gnn_ordering_internal(N, V, E, ad, p);
    greedy_internal(N, V, E, gd, p, color);
}

void gnn_ordering_internal_par(uint32_t N, const uint32_t *V, const uint32_t *E, void *ad, double *p)
{
    struct gnn_data *d = (struct gnn_data *)ad;

    cgnn_graph_sage_16_par(d->params, d->param, N, V, E, d->md, d->x, d->y, p);
}

void gnn_internal_par(uint32_t N, const uint32_t *V, const uint32_t *E,
                      void *ad, void *jp, double *p, int *color)
{
    gnn_ordering_internal_par(N, V, E, ad, p);
    jones_plassmann_internal(N, V, E, jp, p, color);
}

void gnn_ordering(uint32_t N, const uint32_t *V, const uint32_t *E, double *p, int l)
{
    layers = l;
    void *ad = gnn_setup(N, V[N]);
    gnn_ordering_internal_par(N, V, E, ad, p);
    gnn_cleanup(ad);
}

void gnn_runner(uint32_t N, const uint32_t *V, const uint32_t *E, int l,
                int it, int k, int t_tot, int argc, ...)
{
    layers = l;
    char name[256];
    sprintf(name, "GNN-%d", layers);
    va_list argv;
    va_start(argv, argc);
    runner_test_full(N, V, E,
                     gnn_setup,
                     gnn_cleanup,
                     gnn_internal,
                     gnn_internal_par,
                     name, it, k, t_tot, argc, argv);
    va_end(argv);
}