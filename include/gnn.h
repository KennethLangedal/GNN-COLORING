#pragma once
#include <stdint.h>

void gnn_ordering(uint32_t N, const uint32_t *V, const uint32_t *E, double *p, int l);

void gnn_runner(uint32_t N, const uint32_t *V, const uint32_t *E, int l,
                int it, int k, int t_tot, int ntc, const int *nt);