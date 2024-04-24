#pragma once
#include <stdint.h>

// *** Sequential coloring ***

int *greedy_coloring(uint32_t N, const uint32_t *V, const uint32_t *E, const double *p);

void *greedy_setup(uint32_t N, uint32_t M);

void greedy_cleanup(void *gd);

void greedy_internal(uint32_t N, const uint32_t *V, const uint32_t *E,
                     void *gd, const double *p, int *color);

int validate_coloring(uint32_t N, const uint32_t *V, const uint32_t *E, const int *color);

// *** Parallel coloring ***

int *jones_plassmann_coloring(uint32_t N, const uint32_t *V, const uint32_t *E, const double *p);

// N = number of vertices, M = number of edges, md = max degree
void *jones_plassmann_setup(uint32_t N, uint32_t M);

void jones_plassmann_cleanup(void *jp);

void jones_plassmann_internal(uint32_t N, const uint32_t *V, const uint32_t *E,
                              void *jp, const double *p, int *color);

int validate_coloring_par(uint32_t N, const uint32_t *V, const uint32_t *E, const int *color);