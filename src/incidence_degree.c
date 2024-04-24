#include "incidence_degree.h"
#include "coloring.h"

#include <stdlib.h>
#include <omp.h>

static inline int compare(const uint32_t *w, const uint32_t *V, uint32_t i, uint32_t j)
{
    if (w[i] != w[j])
        return w[i] > w[j];
    if (V[i + 1] - V[i] != V[j + 1] - V[j])
        return V[i + 1] - V[i] > V[j + 1] - V[j];
    return i > j;
}

void heap_insert(uint32_t *heap, uint32_t *i_heap, const uint32_t *w, const uint32_t *V, uint32_t *N, uint32_t x)
{
    *N += 1;
    uint32_t i = *N;
    i_heap[x] = i;
    heap[i] = x;

    uint32_t p = i >> 1;
    while (i != 1 && compare(w, V, x, heap[p]))
    {
        uint32_t y = heap[p];
        i_heap[y] = i;
        heap[i] = y;

        i = p;
        i_heap[x] = i;
        heap[i] = x;

        p = i >> 1;
    }
}

void heap_down(uint32_t *heap, uint32_t *i_heap, const uint32_t *w, const uint32_t *V, uint32_t N, uint32_t i)
{
    uint32_t left = i * 2, right = i * 2 + 1, largest = i;

    if (left <= N && compare(w, V, heap[left], heap[largest]))
        largest = left;

    if (right <= N && compare(w, V, heap[right], heap[largest]))
        largest = right;

    if (largest != i)
    {
        uint32_t x = heap[i];
        uint32_t y = heap[largest];
        i_heap[y] = i;
        i_heap[x] = largest;
        heap[i] = y;
        heap[largest] = x;
        heap_down(heap, i_heap, w, V, N, largest);
    }
}

void heap_remove(uint32_t *heap, uint32_t *i_heap, const uint32_t *w, const uint32_t *V, uint32_t *N, uint32_t i)
{
    if (i == 0)
        return;
    uint32_t x = heap[i];
    uint32_t y = heap[*N];

    heap[i] = y;
    heap[*N] = UINT32_MAX;
    i_heap[y] = i;
    i_heap[x] = 0;

    *N -= 1;
    heap_down(heap, i_heap, w, V, *N, i);
}

void explore(const uint32_t *V, const uint32_t *E, uint32_t *cc, uint32_t u, uint32_t t)
{
    if (cc[u] != UINT32_MAX)
        return;

    cc[u] = t;
    for (uint32_t i = V[u]; i < V[u + 1]; i++)
    {
        uint32_t v = E[i];
        explore(V, E, cc, v, t);
    }
}

uint32_t find_cc(uint32_t N, const uint32_t *V, const uint32_t *E, uint32_t *cc)
{
    uint32_t t = 0;
    for (uint32_t i = 0; i < N; i++)
        cc[i] = UINT32_MAX;

    for (uint32_t u = 0; u < N; u++)
    {
        if (cc[u] != UINT32_MAX)
            continue;

        explore(V, E, cc, u, t);
        t++;
    }

    return t;
}

void incidence_degree_ordering(uint32_t N, const uint32_t *V, const uint32_t *E, double *p)
{
    uint32_t n_heap = 0;
    uint32_t *heap = malloc(sizeof(uint32_t) * (N + 1));
    uint32_t *i_heap = malloc(sizeof(uint32_t) * N);
    uint32_t *id = malloc(sizeof(uint32_t) * N);

    for (uint32_t i = 0; i < N + 1; i++)
        heap[i] = UINT32_MAX;

    for (uint32_t i = 0; i < N; i++)
        i_heap[i] = 0;

    uint32_t *cc = malloc(sizeof(int) * N);
    uint32_t n_cc = find_cc(N, V, E, cc);
    uint32_t *md = malloc(sizeof(int) * n_cc);
    for (uint32_t i = 0; i < n_cc; i++)
        md[i] = UINT32_MAX;

    for (uint32_t u = 0; u < N; u++)
    {
        id[u] = 0;
        p[u] = -1.0;

        if (md[cc[u]] == UINT32_MAX || V[md[cc[u]] + 1] - V[md[cc[u]]] < V[u + 1] - V[u])
            md[cc[u]] = u;
    }

    uint32_t t = N;
    heap_insert(heap, i_heap, id, V, &n_heap, md[cc[0]]);

    for (uint32_t k = 0; k < N; k++)
    {
        while (n_heap != 0)
        {
            uint32_t u = heap[1];
            heap_remove(heap, i_heap, id, V, &n_heap, 1);
            p[u] = t;
            for (uint32_t i = V[u]; i < V[u + 1]; i++)
            {
                uint32_t v = E[i];
                if (p[v] >= 0.0)
                    continue;
                id[v]++;
                heap_remove(heap, i_heap, id, V, &n_heap, i_heap[v]);
                heap_insert(heap, i_heap, id, V, &n_heap, v);
            }
            t--;
        }

        if (p[k] < 0.0)
        {
            heap_insert(heap, i_heap, id, V, &n_heap, md[cc[k]]);
        }
    }

    free(heap);
    free(i_heap);
    free(id);
}

void saturation_degree_ordering(uint32_t N, const uint32_t *V, const uint32_t *E, double *p)
{
    uint32_t n_heap = 0;
    uint32_t *heap = malloc(sizeof(uint32_t) * (N + 1));
    uint32_t *i_heap = malloc(sizeof(uint32_t) * N);
    uint32_t *sd = malloc(sizeof(uint32_t) * N);

    for (uint32_t i = 0; i < N + 1; i++)
        heap[i] = INT32_MAX;

    for (uint32_t i = 0; i < N; i++)
        i_heap[i] = 0;

    uint32_t *cc = malloc(sizeof(uint32_t) * N);
    uint32_t n_cc = find_cc(N, V, E, cc);
    uint32_t *md = malloc(sizeof(uint32_t) * n_cc);
    for (uint32_t i = 0; i < n_cc; i++)
        md[i] = UINT32_MAX;

    int *colors = malloc(sizeof(int) * N);
    uint32_t *marks = malloc(sizeof(uint32_t) * N);

    for (uint32_t u = 0; u < N; u++)
    {
        sd[u] = 0;
        p[u] = -1.0;

        colors[u] = 0;
        marks[u] = N;

        if (md[cc[u]] == UINT32_MAX || V[md[cc[u]] + 1] - V[md[cc[u]]] < V[u + 1] - V[u])
            md[cc[u]] = u;
    }

    uint32_t t = N;
    heap_insert(heap, i_heap, sd, V, &n_heap, md[cc[0]]);

    for (uint32_t k = 0; k < N; k++)
    {
        while (n_heap != 0)
        {
            uint32_t u = heap[1];
            heap_remove(heap, i_heap, sd, V, &n_heap, 1);
            p[u] = t;
            t--;

            for (uint32_t i = V[u]; i < V[u + 1]; i++)
                marks[colors[E[i]]] = u;

            int color = 1;
            while (marks[color] == u)
                color++;

            colors[u] = color;

            for (uint32_t i = V[u]; i < V[u + 1]; i++)
            {
                uint32_t v = E[i];
                if (p[v] >= 0.0)
                    continue;

                uint32_t found = 0;
                for (uint32_t j = V[v]; j < V[v + 1]; j++)
                {
                    uint32_t w = E[j];
                    if (w != u && colors[w] == colors[u])
                        found = 1;
                }

                if (!found)
                {
                    sd[v]++;
                    heap_remove(heap, i_heap, sd, V, &n_heap, i_heap[v]);
                    heap_insert(heap, i_heap, sd, V, &n_heap, v);
                }
            }
        }

        if (p[k] < 0.0)
        {
            heap_insert(heap, i_heap, sd, V, &n_heap, md[cc[k]]);
        }
    }

    free(heap);
    free(i_heap);
    free(sd);
    free(cc);
    free(md);
    free(colors);
    free(marks);
}

void saturation_degree_ordering_alt(uint32_t N, const uint32_t *V, const uint32_t *E, double *p)
{
    uint32_t n_heap = 0;
    uint32_t *heap = malloc(sizeof(uint32_t) * (N + 1));
    uint32_t *i_heap = malloc(sizeof(uint32_t) * N);
    uint32_t *sd = malloc(sizeof(uint32_t) * N);

    for (uint32_t i = 0; i < N + 1; i++)
        heap[i] = UINT32_MAX;

    for (uint32_t i = 0; i < N; i++)
        i_heap[i] = 0;

    uint32_t *cc = malloc(sizeof(uint32_t) * N);
    uint32_t n_cc = find_cc(N, V, E, cc);
    uint32_t *md = malloc(sizeof(uint32_t) * n_cc);
    for (uint32_t i = 0; i < n_cc; i++)
        md[i] = 0;

    int *colors = malloc(sizeof(int) * N);
    uint32_t *marks = malloc(sizeof(uint32_t) * N);

    uint32_t *buffer = malloc(sizeof(uint32_t) * N);

    for (uint32_t u = 0; u < N; u++)
    {
        sd[u] = 0;
        p[u] = -1.0;

        colors[u] = 0;
        marks[u] = N;

        if (md[cc[u]] < V[u + 1] - V[u])
            md[cc[u]] = V[u + 1] - V[u];
    }

    uint32_t t = N;

    for (uint32_t u = 0; u < N; u++)
        if (V[u + 1] - V[u] == md[cc[u]])
            heap_insert(heap, i_heap, sd, V, &n_heap, u);

    while (n_heap != 0)
    {
        uint32_t b = 0;
        uint32_t best_sd = sd[heap[1]];
        uint32_t best_degree = V[heap[1] + 1] - V[heap[1]];
        while (n_heap > 0 && sd[heap[1]] == best_sd && (V[heap[1] + 1] - V[heap[1]]) == best_degree)
        {
            uint32_t u = heap[1];
            heap_remove(heap, i_heap, sd, V, &n_heap, 1);
            p[u] = t;
            buffer[b++] = u;
        }
        t--;

        for (uint32_t bi = 0; bi < b; bi++)
        {
            uint32_t u = buffer[bi];
            for (uint32_t i = V[u]; i < V[u + 1]; i++)
                marks[colors[E[i]]] = u;

            int color = 1;
            while (marks[color] == u)
                color++;

            colors[u] = color;

            for (uint32_t i = V[u]; i < V[u + 1]; i++)
            {
                uint32_t v = E[i];
                if (p[v] >= 0.0)
                    continue;

                uint32_t found = 0;
                for (uint32_t j = V[v]; j < V[v + 1]; j++)
                {
                    uint32_t w = E[j];
                    if (w != u && colors[w] == colors[u])
                        found = 1;
                }

                if (!found)
                {
                    sd[v]++;
                    heap_remove(heap, i_heap, sd, V, &n_heap, i_heap[v]);
                    heap_insert(heap, i_heap, sd, V, &n_heap, v);
                }
            }
        }
    }

    free(heap);
    free(i_heap);
    free(sd);
    free(cc);
    free(md);
    free(colors);
    free(marks);
    free(buffer);
}