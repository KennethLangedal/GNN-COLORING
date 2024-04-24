#pragma once
#include <stdint.h>

void largest_degree_first_ordering(uint32_t N, const uint32_t *V, const uint32_t *E, double *p);

int *largest_degree_first_coloring(uint32_t N, const uint32_t *V, const uint32_t *E);

int *largest_degree_first_coloring_par(uint32_t N, const uint32_t *V, const uint32_t *E);

void largest_degree_first_runner(uint32_t N, const uint32_t *V, const uint32_t *E,
                                 int it, int k, int t_tot, int ntc, const int *nt);

void largest_log_degree_first_ordering(uint32_t N, const uint32_t *V, const uint32_t *E, double *p);

int *largest_log_degree_first_coloring(uint32_t N, const uint32_t *V, const uint32_t *E);

int *largest_log_degree_first_coloring_par(uint32_t N, const uint32_t *V, const uint32_t *E);

void largest_log_degree_first_runner(uint32_t N, const uint32_t *V, const uint32_t *E,
                                     int it, int k, int t_tot, int ntc, const int *nt);