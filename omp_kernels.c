#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_N ((size_t)1 << 24)
#define DEFAULT_GEMM_N ((size_t)1024)
#define DEFAULT_TILE_SIZE ((size_t)64)
#define GEMM_TIMED_RUNS 3

typedef void (*gemm_function)(const double *, const double *, double *,
                              size_t, size_t);

static int parse_length(const char *text, size_t *length)
{
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno != 0 || text[0] == '-' || end == text || *end != '\0' ||
        parsed == 0 || parsed > SIZE_MAX ||
        parsed > SIZE_MAX / sizeof(double)) {
        return 0;
    }
    *length = (size_t)parsed;
    return 1;
}

static void print_result(const char *name, double result, double expected,
                         double elapsed_seconds)
{
    printf("%-12s result = %.17g, absolute error = %.17g, time = %.6f s\n",
           name, result, fabs(result - expected), elapsed_seconds);
}

static void gemm_naive(const double *a, const double *b, double *c, size_t n,
                       size_t unused_tile_size)
{
    (void)unused_tile_size;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

static void gemm_parallel(const double *a, const double *b, double *c,
                          size_t n, size_t unused_tile_size)
{
    (void)unused_tile_size;
    // Parallelize rows: each thread writes distinct C rows; A and B are read-only.
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

static void gemm_tiled(const double *a, const double *b, double *c, size_t n,
                       size_t tile_size)
{
    memset(c, 0, n * n * sizeof(*c));

    // Each collapsed iteration owns one non-overlapping tile of C.
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t ii = 0; ii < n; ii += tile_size) {
        for (size_t jj = 0; jj < n; jj += tile_size) {
            size_t i_end = ii + tile_size < n ? ii + tile_size : n;
            size_t j_end = jj + tile_size < n ? jj + tile_size : n;
            for (size_t kk = 0; kk < n; kk += tile_size) {
                size_t k_end = kk + tile_size < n ? kk + tile_size : n;
                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t k = kk; k < k_end; ++k) {
                        double a_value = a[i * n + k];
                        for (size_t j = jj; j < j_end; ++j) {
                            c[i * n + j] += a_value * b[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

static double benchmark_gemm(gemm_function function, const double *a,
                             const double *b, double *c, size_t n,
                             size_t tile_size)
{
    function(a, b, c, n, tile_size);

    double total_seconds = 0.0;
    for (int run = 0; run < GEMM_TIMED_RUNS; ++run) {
        double start = omp_get_wtime();
        function(a, b, c, n, tile_size);
        total_seconds += omp_get_wtime() - start;
    }
    return total_seconds / GEMM_TIMED_RUNS;
}

static double max_absolute_difference(const double *left, const double *right,
                                      size_t count)
{
    double max_difference = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double difference = fabs(left[i] - right[i]);
        if (difference > max_difference) {
            max_difference = difference;
        }
    }
    return max_difference;
}

static int run_gemm(size_t n, size_t tile_size)
{
    if (n > SIZE_MAX / n || n * n > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "Matrix size is too large\n");
        return EXIT_FAILURE;
    }
    size_t matrix_elements = n * n;
    size_t matrix_bytes = matrix_elements * sizeof(double);
    double *a = malloc(matrix_bytes);
    double *b = malloc(matrix_bytes);
    double *reference = malloc(matrix_bytes);
    double *parallel = malloc(matrix_bytes);
    double *tiled = malloc(matrix_bytes);
    if (a == NULL || b == NULL || reference == NULL || parallel == NULL ||
        tiled == NULL) {
        fprintf(stderr, "Could not allocate matrices of size %zu x %zu\n", n,
                n);
        free(a);
        free(b);
        free(reference);
        free(parallel);
        free(tiled);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < matrix_elements; ++i) {
        a[i] = (double)(i % 17) / 17.0;
        b[i] = (double)(i % 19) / 19.0;
    }

    double serial_seconds =
        benchmark_gemm(gemm_naive, a, b, reference, n, tile_size);
    double parallel_seconds =
        benchmark_gemm(gemm_parallel, a, b, parallel, n, tile_size);
    double tiled_seconds =
        benchmark_gemm(gemm_tiled, a, b, tiled, n, tile_size);

    double parallel_difference =
        max_absolute_difference(reference, parallel, matrix_elements);
    double tiled_difference =
        max_absolute_difference(reference, tiled, matrix_elements);
    double operations = 2.0 * (double)n * (double)n * (double)n;

    printf("GEMM: N = %zu, tile size = %zu, OpenMP max threads = %d\n", n,
           tile_size, omp_get_max_threads());
    printf("Timing: one warm-up and %d timed runs per version\n",
           GEMM_TIMED_RUNS);
    printf("%-18s average = %.6f s, %.3f GFLOP/s\n", "naive serial",
           serial_seconds, operations / serial_seconds / 1.0e9);
    printf("%-18s average = %.6f s, %.3f GFLOP/s, max diff = %.3g\n",
           "naive OpenMP", parallel_seconds,
           operations / parallel_seconds / 1.0e9, parallel_difference);
    printf("%-18s average = %.6f s, %.3f GFLOP/s, max diff = %.3g\n",
           "tiled OpenMP", tiled_seconds,
           operations / tiled_seconds / 1.0e9, tiled_difference);

    free(a);
    free(b);
    free(reference);
    free(parallel);
    free(tiled);
    return parallel_difference < 1.0e-9 && tiled_difference < 1.0e-9
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}

int main(int argc, char **argv)
{
    if (argc >= 2 && strcmp(argv[1], "gemm") == 0) {
        size_t n = DEFAULT_GEMM_N;
        size_t tile_size = DEFAULT_TILE_SIZE;
        if (argc > 4 ||
            (argc >= 3 && !parse_length(argv[2], &n)) ||
            (argc == 4 && !parse_length(argv[3], &tile_size))) {
            fprintf(stderr,
                    "Usage: %s gemm [matrix_size [tile_size]]\n", argv[0]);
            return EXIT_FAILURE;
        }
        return run_gemm(n, tile_size);
    }

    size_t n = DEFAULT_N;
    size_t synchronized_n = n;
    if (argc > 3) {
        fprintf(stderr,
                "Usage: %s [vector_length [critical_atomic_length]]\n",
                argv[0]);
        return EXIT_FAILURE;
    }
    if (argc >= 2) {
        if (!parse_length(argv[1], &n)) {
            fprintf(stderr, "vector_length must be a positive valid length\n");
            return EXIT_FAILURE;
        }
    }
    synchronized_n = n;
    if (argc == 3) {
        if (!parse_length(argv[2], &synchronized_n) || synchronized_n > n) {
            fprintf(stderr,
                    "critical_atomic_length must be positive and no larger "
                    "than vector_length\n");
            return EXIT_FAILURE;
        }
    }

    double *a = malloc(n * sizeof(*a));
    double *b = malloc(n * sizeof(*b));
    if (a == NULL || b == NULL) {
        fprintf(stderr, "Could not allocate vectors of length %zu\n", n);
        free(a);
        free(b);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < n; ++i) {
        a[i] = 1.0;
        b[i] = 0.5;
    }

    const double expected = (double)n / 2.0;
    const double synchronized_expected = (double)synchronized_n / 2.0;
    double sum;
    double start;

    printf("Vector length: %zu; critical/atomic length: %zu; "
           "OpenMP max threads: %d\n",
           n, synchronized_n, omp_get_max_threads());

    // sum = 0.0;
    // start = omp_get_wtime();
    // #pragma omp parallel for
    // for (size_t i = 0; i < n; ++i) {
    //     sum += a[i] * b[i];
    // }
    // print_result("racy", sum, expected, omp_get_wtime() - start);

    // sum = 0.0;
    // start = omp_get_wtime();
    // #pragma omp parallel for
    // for (size_t i = 0; i < synchronized_n; ++i) {
    //     double product = a[i] * b[i];
    //     #pragma omp critical
    //     {
    //         sum += product;
    //     }
    // }
    // print_result("critical", sum, synchronized_expected,
    //              omp_get_wtime() - start);

    // sum = 0.0;
    // start = omp_get_wtime();
    // #pragma omp parallel for
    // for (size_t i = 0; i < synchronized_n; ++i) {
    //     #pragma omp atomic update
    //     sum += a[i] * b[i];
    // }
    // print_result("atomic", sum, synchronized_expected,
    //              omp_get_wtime() - start);

    // sum = 0.0;
    // start = omp_get_wtime();
    // #pragma omp parallel for reduction(+ : sum)
    // for (size_t i = 0; i < n; ++i) {
    //     sum += a[i] * b[i];
    // }
    // print_result("reduction", sum, expected, omp_get_wtime() - start);

    printf("All done!\n");
    free(a);
    free(b);
    return EXIT_SUCCESS;
}
