#pragma once
#include <stdio.h>
#include <stdint.h>

/*
    Matrix Market file format:

    %%MatrixMarket matrix coordinate real general
    % <- comments
    M N L
    I1 J1 A(I1, J1)
    I2 J2 A(I2, J2)
    I3 J3 A(I3, J3)
    . . .
    IL JL A(IL, JL)

    ---------------

    Input comments:
        - M, N, and L are rows, columns, and nonzeros
        - 1 based indexing
        - The first 3 header arguments are fixed, then:
            - real/complex/integer/patter (if pattern then A == NULL)
            - general/symmetric

    ---------------

    After parsing:
        - Converted to 0 based index
        - Made undirected (could duplicate edges)
        - A values ignored
        - N = max(M, N)
*/

void mtx_parse(FILE *f, uint32_t *N, uint32_t **V, uint32_t **E);

void mtx_parse_par(FILE *f, uint32_t *N, uint32_t **V, uint32_t **E);

void mtx_store(FILE *f, uint32_t N, const uint32_t *V, const uint32_t *E);

/*
    Adjacency graph format used by GBBS (among others)

    AdjacencyGraph
    N
    M
    offset 0
    offset 1
    . . .
    offset N - 1
    edge 0
    edge 1
    . . .
    edge M - 1

    ---------------

    Input comments:
        - N is number of vertices and M number of edges
        - 0 based indexing
        - Offsets means sections of edges

    ---------------

    After parsing:
        - Not made undirected, could have directed edges
*/

void adj_parse(FILE *f, uint32_t *N, uint32_t **V, uint32_t **E);

void adj_store(FILE *f, uint32_t N, const uint32_t *V, const uint32_t *E);

/*
    .gr file format used in PACE (among others)

    c <- comments
    p td N M
    u1 v1
    u2 v2
    u3 v3
    . . .
    uM vM

    ---------------

    Input comments:
        - p header line prefix
        - td is problem specific (anything is ok as long as it's just text)
        - N is number of vertices and M number of edges
        - 1 based indexing

    ---------------

    After parsing:
        - Converted to 0 based index
        - Made undirected (could duplicate edges)
*/

void gr_parse(FILE *f, uint32_t *N, uint32_t **V, uint32_t **E);

void gr_store(FILE *f, uint32_t N, const uint32_t *V, const uint32_t *E);

/*
    .graph file format used by METIS (among others)

    % <- comments
    N M edge_w/node_w
    N(u1)
    N(u2)
    ..
    N(uN)

    ---------------

    After parsing:
        - Converted to 0 based index
        - Made undirected (could duplicate edges)
*/

void metis_parse(FILE *f, uint32_t *N, uint32_t **V, uint32_t **E, uint32_t **W);

void metis_store(FILE *f, uint32_t N, const uint32_t *V, const uint32_t *E, const uint32_t *W);

/*
    Utility functions for graphs
*/

void graph_sort_edges(uint32_t N, const uint32_t *V, uint32_t *E);

void graph_sort_edges_par(uint32_t N, const uint32_t *V, uint32_t *E);

// Assumes undirected and sorted edges, will only delete edges
void graph_make_simple(uint32_t N, uint32_t *V, uint32_t **E);

/*
    Returns 0 if any of the following is true:
        - Errors in V
        - Errors in E
        - Self edge
        - Multi edge
        - Edges not sorted
        - Directed edge
        - Incorrect number of edges
    Returns 1 otherwise
*/
uint32_t graph_validate(uint32_t N, const uint32_t *V, const uint32_t *E);

uint32_t graph_validate_par(uint32_t N, const uint32_t *V, const uint32_t *E);