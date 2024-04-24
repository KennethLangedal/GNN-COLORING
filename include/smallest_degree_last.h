#pragma once
#include <stdint.h>

void smallest_degree_last_ordering(uint32_t N, const uint32_t *V, const uint32_t *E, double *p);

int *smallest_degree_last_coloring(uint32_t N, const uint32_t *V, const uint32_t *E);

int *smallest_degree_last_coloring_par(uint32_t N, const uint32_t *V, const uint32_t *E);

void smallest_degree_last_runner(uint32_t N, const uint32_t *V, const uint32_t *E,
                                 int it, int k, int t_tot, int argc, ...);

void smallest_log_degree_last_ordering(uint32_t N, const uint32_t *V, const uint32_t *E, double *p);

int *smallest_log_degree_last_coloring(uint32_t N, const uint32_t *V, const uint32_t *E);

int *smallest_log_degree_last_coloring_par(uint32_t N, const uint32_t *V, const uint32_t *E);

void smallest_log_degree_last_runner(uint32_t N, const uint32_t *V, const uint32_t *E,
                                     int it, int k, int t_tot, int argc, ...);
