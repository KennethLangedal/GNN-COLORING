#include "fast_io.h"

#include <omp.h>
#include <stdlib.h>
#include <sys/mman.h>

static inline void parse_id(char *data, size_t *p, uint32_t *v)
{
    while (data[*p] < '0' || data[*p] > '9')
        (*p)++;

    *v = 0;
    while (data[*p] >= '0' && data[*p] <= '9')
        *v = (*v) * 10 + data[(*p)++] - '0';
}

static inline void skip_line(char *data, size_t *p)
{
    while (data[*p] != '\n')
        (*p)++;
    (*p)++;
}

static inline void skip_line_safe(char *data, size_t *p, size_t t)
{
    while (*p < t && data[*p] != '\n')
        (*p)++;
    (*p)++;
}

void mtx_parse_raw(FILE *f, uint32_t *N, uint32_t *L, uint32_t **I, uint32_t **J)
{
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *data = mmap(0, size, PROT_READ, MAP_PRIVATE, fileno_unlocked(f), 0);
    size_t p = 0;

    while (data[p] == '%')
        skip_line(data, &p);

    uint32_t M;

    parse_id(data, &p, &M);
    parse_id(data, &p, N);
    parse_id(data, &p, L);

    if (M > *N)
        *N = M;

    *I = malloc(sizeof(uint32_t) * (*L));
    *J = malloc(sizeof(uint32_t) * (*L));

    for (uint32_t i = 0; i < (*L); i++)
    {
        skip_line(data, &p);
        while (data[p] == '%')
            skip_line(data, &p);

        parse_id(data, &p, (*I) + i);
        parse_id(data, &p, (*J) + i);
    }

    munmap(data, size);
}

void mtx_parse_raw_par(FILE *f, uint32_t *N, uint32_t *L, uint32_t **I, uint32_t **J)
{
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *data = mmap(0, size, PROT_READ, MAP_SHARED, fileno_unlocked(f), 0);
    size_t p = 0;

    while (data[p] == '%')
        skip_line(data, &p);

    uint32_t M;

    parse_id(data, &p, &M);
    parse_id(data, &p, N);
    parse_id(data, &p, L);

    if (M > *N)
        *N = M;

    *I = malloc(sizeof(uint32_t) * (*L));
    *J = malloc(sizeof(uint32_t) * (*L));

    uint32_t *tc;

#pragma omp parallel firstprivate(p)
    {
        uint32_t tid = omp_get_thread_num();
        uint32_t nt = omp_get_num_threads();

        size_t s = ((size - p) / nt) * tid + p;
        size_t t = s + (size - p) / nt;
        if (tid == nt - 1)
            t = size;

        if (tid == 0)
            tc = malloc(sizeof(uint32_t) * nt);

#pragma omp barrier

        uint32_t lc = 0;
        for (size_t i = s; i < t; i++)
            if (data[i] == '\n')
                lc++;
        tc[tid] = lc;

#pragma omp barrier

        p = s;
        s = 0;
        for (uint32_t i = 0; i < tid; i++)
            s += tc[i];

        t = s + tc[tid];
        if (tid == nt - 1 || t > *L)
            t = *L;

        for (uint32_t i = s; i < t; i++)
        {
            skip_line_safe(data, &p, size);

            parse_id(data, &p, (*I) + i);
            parse_id(data, &p, (*J) + i);
        }

#pragma omp barrier

        if (tid == 0)
            free(tc);
    }

    munmap(data, size);
}

void mtx_to_graph(uint32_t L, uint32_t *I, uint32_t *J, uint32_t N, uint32_t **V, uint32_t **E)
{
    uint32_t M = L * 2;

    *V = malloc(sizeof(uint32_t) * (N + 1));
    for (uint32_t i = 0; i < N + 1; i++)
        (*V)[i] = 0;

    for (uint32_t i = 0; i < L; i++)
    {
        (*V)[I[i] - 1]++;
        (*V)[J[i] - 1]++;
    }

    for (uint32_t i = 1; i <= N; i++)
        (*V)[i] += (*V)[i - 1];

    *E = malloc(sizeof(uint32_t) * M);

    for (uint32_t i = 0; i < L; i++)
    {
        (*V)[I[i] - 1]--;
        (*E)[(*V)[I[i] - 1]] = J[i] - 1;
        (*V)[J[i] - 1]--;
        (*E)[(*V)[J[i] - 1]] = I[i] - 1;
    }
}

void mtx_to_graph_par(uint32_t L, uint32_t *I, uint32_t *J, uint32_t N, uint32_t **V, uint32_t **E)
{
    uint32_t M = L * 2;

    *V = malloc(sizeof(uint32_t) * (N + 1));
#pragma omp parallel for
    for (uint32_t i = 0; i < N + 1; i++)
        (*V)[i] = 0;

        // Count degree
#pragma omp parallel for
    for (uint32_t i = 0; i < L; i++)
    {
        __atomic_add_fetch((*V) + (I[i] - 1), 1, __ATOMIC_RELAXED);
        __atomic_add_fetch((*V) + (J[i] - 1), 1, __ATOMIC_RELAXED);
    }

    for (uint32_t i = 1; i <= N; i++)
        (*V)[i] += (*V)[i - 1];

    *E = malloc(sizeof(uint32_t) * M);

#pragma omp parallel for
    for (uint32_t i = 0; i < L; i++)
    {
        uint32_t j = __atomic_sub_fetch((*V) + (I[i] - 1), 1, __ATOMIC_RELAXED);
        (*E)[j] = J[i] - 1;

        j = __atomic_sub_fetch((*V) + (J[i] - 1), 1, __ATOMIC_RELAXED);
        (*E)[j] = I[i] - 1;
    }
}

void mtx_parse(FILE *f, uint32_t *N, uint32_t **V, uint32_t **E)
{
    uint32_t *I, *J, L;
    mtx_parse_raw(f, N, &L, &I, &J);
    mtx_to_graph(L, I, J, *N, V, E);

    free(I);
    free(J);
}

void mtx_parse_par(FILE *f, uint32_t *N, uint32_t **V, uint32_t **E)
{
    uint32_t *I, *J, L;
    mtx_parse_raw_par(f, N, &L, &I, &J);
    mtx_to_graph_par(L, I, J, *N, V, E);

    free(I);
    free(J);
}

void mtx_store(FILE *f, uint32_t N, const uint32_t *V, const uint32_t *E)
{
    fprintf(f, "%%MatrixMarket matrix coordinate patter symmetric\n");
    fprintf(f, "%d %d %d\n", N, N, V[N] / 2);
    for (uint32_t u = 0; u < N; u++)
    {
        for (uint32_t i = V[u]; i < V[u + 1]; i++)
        {
            uint32_t v = E[i];
            if (v > u)
                fprintf(f, "%d %d\n", u + 1, v + 1);
        }
    }
}

void adj_parse(FILE *f, uint32_t *N, uint32_t **V, uint32_t **E)
{
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *data = mmap(0, size, PROT_READ, MAP_PRIVATE, fileno_unlocked(f), 0);
    size_t i = 0;

    uint32_t M;
    parse_id(data, &i, N);
    parse_id(data, &i, &M);

    *V = malloc(sizeof(uint32_t) * ((*N) + 1));

    for (uint32_t j = 0; j < *N; j++)
        parse_id(data, &i, (*V) + j);

    (*V)[*N] = M;

    *E = malloc(sizeof(uint32_t) * M);
    for (uint32_t j = 0; j < M; j++)
        parse_id(data, &i, (*E) + j);

    munmap(data, size);
}

void adj_store(FILE *f, uint32_t N, const uint32_t *V, const uint32_t *E)
{
    fprintf(f, "AdjacencyGraph\n");
    fprintf(f, "%d\n", N);
    fprintf(f, "%d\n", V[N]);

    for (uint32_t i = 0; i < N; i++)
        fprintf(f, "%d\n", V[i]);

    for (uint32_t i = 0; i < V[N]; i++)
        fprintf(f, "%d\n", E[i]);
}

struct gr
{
    uint32_t M, N;
    uint32_t *I, *J;
};

struct gr parse_raw_gr(FILE *f)
{
    struct gr m;

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *data = mmap(0, size, PROT_READ, MAP_PRIVATE, fileno_unlocked(f), 0);
    size_t i = 0;

    while (data[i] == 'c')
        skip_line(data, &i);

    parse_id(data, &i, &m.N);
    parse_id(data, &i, &m.M);

    m.I = malloc(sizeof(uint32_t) * m.M);
    m.J = malloc(sizeof(uint32_t) * m.M);

    uint32_t x, y;
    for (uint32_t j = 0; j < m.M; j++)
    {
        skip_line(data, &i);
        while (data[i] == 'c')
            skip_line(data, &i);

        parse_id(data, &i, &x);
        parse_id(data, &i, &y);
        m.I[j] = x - 1;
        m.J[j] = y - 1;
    }

    munmap(data, size);

    return m;
}

void free_gr(struct gr m)
{
    free(m.I);
    free(m.J);
}

void gr_parse(FILE *f, uint32_t *N, uint32_t **V, uint32_t **E)
{
    struct gr m = parse_raw_gr(f);
    *N = m.N;
    *V = malloc(sizeof(uint32_t) * ((*N) + 1));

    for (uint32_t i = 0; i < (*N) + 1; i++)
        (*V)[i] = 0;

    for (uint32_t i = 0; i < m.M; i++)
    {
        (*V)[m.I[i]]++;
        (*V)[m.J[i]]++;
    }

    for (uint32_t i = 1; i <= (*N); i++)
        (*V)[i] += (*V)[i - 1];

    *E = malloc(sizeof(uint32_t) * m.M * 2);

    for (uint32_t i = 0; i < m.M; i++)
    {
        (*V)[m.I[i]]--;
        (*E)[(*V)[m.I[i]]] = m.J[i];
        (*V)[m.J[i]]--;
        (*E)[(*V)[m.J[i]]] = m.I[i];
    }

    free_gr(m);
}

void gr_store(FILE *f, uint32_t N, const uint32_t *V, const uint32_t *E)
{
    fprintf(f, "p td %d %d\n", N, V[N] / 2);
    for (uint32_t u = 0; u < N; u++)
    {
        for (uint32_t i = V[u]; i < V[u + 1]; i++)
        {
            uint32_t v = E[i];
            if (v > u)
                fprintf(f, "%d %d\n", u + 1, v + 1);
        }
    }
}

void metis_parse(FILE *f, uint32_t *N, uint32_t **V, uint32_t **E, uint32_t **W)
{
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *data = mmap(0, size, PROT_READ, MAP_PRIVATE, fileno_unlocked(f), 0);
    size_t i = 0;

    while (data[i] == '%')
        skip_line(data, &i);

    uint32_t M, p;
    parse_id(data, &i, N);
    parse_id(data, &i, &M);
    parse_id(data, &i, &p);

    *V = malloc(sizeof(uint32_t) * (*N + 1));
    *E = malloc(sizeof(uint32_t) * M * 2);
    *W = (p == 10 || p == 11) ? malloc(sizeof(uint32_t) * *N) : NULL;

    (*V)[0] = 0;
    uint32_t w, v;
    for (uint32_t u = 0; u < *N; u++)
    {
        skip_line(data, &i);
        while (data[i] == '%')
            skip_line(data, &i);

        if (p == 10 || p == 11)
            parse_id(data, &i, &(*W)[u]);

        uint32_t s = (*V)[u];
        while (data[i] != '\n' && s < M * 2)
        {
            parse_id(data, &i, &v);
            (*E)[s++] = v - 1;
            if (p == 1 || p == 11)
                parse_id(data, &i, &w);

            while (data[i] == ' ' && s < M * 2)
                i++;
        }
        (*V)[u + 1] = s;
    }

    munmap(data, size);
}

void metis_store(FILE *f, uint32_t N, const uint32_t *V, const uint32_t *E, const uint32_t *W)
{
    fprintf(f, "%d %d %d\n", N, V[N] / 2, W == NULL ? 0 : 10);
    for (uint32_t u = 0; u < N; u++)
    {
        if (W != NULL)
            fprintf(f, "%d ", W[u]);
        for (uint32_t i = V[u]; i < V[u + 1]; i++)
        {
            uint32_t v = E[i];
            fprintf(f, "%d ", v + 1);
        }
        fprintf(f, "\n");
    }
}

static inline int compare(const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}

void graph_sort_edges(uint32_t N, const uint32_t *V, uint32_t *E)
{
    for (uint32_t i = 0; i < N; i++)
        qsort(E + V[i], V[i + 1] - V[i], sizeof(uint32_t), compare);
}

void graph_sort_edges_par(uint32_t N, const uint32_t *V, uint32_t *E)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < N; i++)
        qsort(E + V[i], V[i + 1] - V[i], sizeof(uint32_t), compare);
}

void graph_make_simple(uint32_t N, uint32_t *V, uint32_t **E)
{
    uint32_t i = 0, r = 0;
    for (uint32_t u = 0; u < N; u++)
    {
        uint32_t _r = r;
        for (uint32_t j = V[u]; j < V[u + 1]; j++)
        {
            uint32_t v = (*E)[j];
            if (v == u || (j > V[u] && v <= (*E)[j - 1]))
                r++;
            else
                (*E)[i++] = v;
        }
        V[u] -= _r;
    }
    V[N] -= r;

    if (r > 0)
        *E = realloc(*E, sizeof(uint32_t) * V[N]);
}

uint32_t graph_validate(uint32_t N, const uint32_t *V, const uint32_t *E)
{
    uint32_t M = 0;
    for (uint32_t u = 0; u < N; u++)
    {
        if (V[u + 1] - V[u] < 0)
            return 0;

        M += V[u + 1] - V[u];

        for (uint32_t i = V[u]; i < V[u + 1]; i++)
        {
            if (i < 0 || i >= V[N])
                return 0;

            uint32_t v = E[i];
            if (v < 0 || v >= N || v == u || (i > V[u] && v <= E[i - 1]))
                return 0;

            if (bsearch(&u, E + V[v], V[v + 1] - V[v], sizeof(uint32_t), compare) == NULL)
                return 0;
        }
    }

    if (M != V[N])
        return 0;

    return 1;
}

uint32_t graph_validate_par(uint32_t N, const uint32_t *V, const uint32_t *E)
{
    uint32_t M = 0;
    uint32_t error = 0;

#pragma omp parallel for reduction(+ : M, error)
    for (uint32_t u = 0; u < N; u++)
    {
        if (V[u + 1] - V[u] < 0)
            error = 1;

        M += V[u + 1] - V[u];

        for (uint32_t i = V[u]; i < V[u + 1]; i++)
        {
            if (i < 0 || i >= V[N])
                error = 1;

            uint32_t v = E[i];
            if (v < 0 || v >= N || v == u || (i > V[u] && v <= E[i - 1]))
                error = 1;

            if (bsearch(&u, E + V[v], V[v + 1] - V[v], sizeof(uint32_t), compare) == NULL)
                error = 1;
        }
    }

    if (M != V[N])
        error = 1;

    return error == 0;
}
