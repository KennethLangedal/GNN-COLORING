#pragma once
#include <stdint.h>

void first_fit_ordering(uint32_t N, const uint32_t *V, const uint32_t *E, double *p);

void first_fit_ordering_par(uint32_t N, const uint32_t *V, const uint32_t *E, double *p);

int *first_fit_coloring(uint32_t N, const uint32_t *V, const uint32_t *E);

int *first_fit_coloring_par(uint32_t N, const uint32_t *V, const uint32_t *E);

void first_fit_runner(uint32_t N, const uint32_t *V, const uint32_t *E,
                      int it, int k, int t_tot, int argc, ...);