#pragma once
#include <stdint.h>

/*
    Specialized functions for graph sage with 16 features
    The inputs features are degree and ID, constructed in registers.
    Where the first 8 features being degree / max degree, and the following 8 are ID / N.
    The output is a single element per vertex equal to the sum of its output features.

    x, y, and p should be allocated with 8 extra rows for vectorization.
*/

void cgnn_graph_sage_16(int params, const float *param,
                        uint32_t N, const uint32_t *V, const uint32_t *E, int md,
                        float *x, float *y, double *p);

void cgnn_graph_sage_16_par(int params, const float *param,
                            uint32_t N, const uint32_t *V, const uint32_t *E, int md,
                            float *x, float *y, double *p);