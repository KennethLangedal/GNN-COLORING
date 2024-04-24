#pragma once

#include <stdarg.h>
#include <stdint.h>

void runner_test_ordering(uint32_t N, const uint32_t *V, const uint32_t *E, const char *name, int k,
                          void (*f)(uint32_t, const uint32_t *, const uint32_t *, double *));

void runner_test_ordering_time(uint32_t N, const uint32_t *V, const uint32_t *E, const char *name,
                               void (*f)(uint32_t, const uint32_t *, const uint32_t *, double *));

void runner_test_full(uint32_t N, const uint32_t *V, const uint32_t *E,
                      void *(*setup)(uint32_t, uint32_t),
                      void (*cleanup)(void *),
                      void (*f_seq)(uint32_t, const uint32_t *, const uint32_t *, void *, void *, double *, int *),
                      void (*f_par)(uint32_t, const uint32_t *, const uint32_t *, void *, void *, double *, int *),
                      const char *s, int it, int k, int t_tot, int ntc, const int *nt);