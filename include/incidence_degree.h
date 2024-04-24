#pragma once
#include <stdint.h>

void incidence_degree_ordering(uint32_t N, const uint32_t *V, const uint32_t *E, double *p);

void saturation_degree_ordering(uint32_t N, const uint32_t *V, const uint32_t *E, double *p);

void saturation_degree_ordering_alt(uint32_t N, const uint32_t *V, const uint32_t *E, double *p);