#include <errno.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DATA_SIZE ((size_t)1 << 24)
#define DEFAULT_GEMM_N ((size_t)1024)
#define TILE_SIZE ((size_t)64)
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

static void print_result(const char *name, double result, double expected_sum, double elapsed_seconds)
{
    double absolute_error = (result - expected_sum);
    if (absolute_error < 0.0) {
        absolute_error = -absolute_error;
    }
    printf("%-12s result = %.17g, absolute error = %.17g, time = %.6f s\n",
           name, result, absolute_error, elapsed_seconds);
}

static void gemm_naive(const double *a, const double *b, double *c, size_t n)
{
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
                          size_t n)
{
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

static double benchmark_gemm(const char *name, const double *a, const double *b, double *c, size_t n, size_t tile_size)
{
    
    if (strcmp(name, "gemm_naive") == 0) {
        gemm_naive(a, b, c, n);
    } else if (strcmp(name, "gemm_parallel") == 0) {
        gemm_parallel(a, b, c, n);
    } else if (strcmp(name, "gemm_tiled") == 0) {
        gemm_tiled(a, b, c, n, tile_size);
    } else {
        fprintf(stderr, "Unknown GEMM function name: %s\n", name);
        return -1.0;
    }


    double total_seconds = 0.0;
    for (int run = 0; run < GEMM_TIMED_RUNS; ++run) {
        double start = omp_get_wtime();
        if (strcmp(name, "gemm_naive") == 0) {
            gemm_naive(a, b, c, n);
        } else if (strcmp(name, "gemm_parallel") == 0) {
            gemm_parallel(a, b, c, n);
        } else if (strcmp(name, "gemm_tiled") == 0) {
            gemm_tiled(a, b, c, n, tile_size);
        }
        double time_diff = omp_get_wtime() - start;
        total_seconds += time_diff;
    }
    return total_seconds / GEMM_TIMED_RUNS;
}

static double reduction_dot(const double *a, const double *b, size_t n)
{
    double sum = 0.0;
    #pragma omp parallel for reduction(+ : sum) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

static double benchmark_reduction_dot(const double *a, const double *b,
                                      size_t n)
{
    (void)reduction_dot(a, b, n);

    double total_seconds = 0.0;
    for (int run = 0; run < GEMM_TIMED_RUNS; ++run) {
        double start = omp_get_wtime();
        double result = reduction_dot(a, b, n);
        total_seconds += omp_get_wtime() - start;
        if (fabs(result - (double)n / 2.0) >= 1.0e-9) {
            fprintf(stderr, "Reduction dot product produced an incorrect result\n");
            return -1.0;
        }
    }
    return total_seconds / GEMM_TIMED_RUNS;
}

static int actual_openmp_threads(void)
{
    int actual_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }
    return actual_threads;
}

static size_t make_thread_counts(int *thread_counts, size_t capacity,
                                 int logical_processors)
{
    size_t count = 0;
    int threads = 1;
    while (threads <= logical_processors && count < capacity) {
        thread_counts[count++] = threads;
        if (threads > logical_processors / 2) {
            break;
        }
        threads *= 2;
    }
    if (thread_counts[count - 1] != logical_processors &&
        count < capacity) {
        thread_counts[count++] = logical_processors;
    }
    return count;
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
        return 1;
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
        return 1;
    }

    for (size_t i = 0; i < matrix_elements; ++i) {
        a[i] = (double)(i % 17) / 17.0;
        b[i] = (double)(i % 19) / 19.0;
    }

    double serial_seconds =
        benchmark_gemm("gemm_naive", a, b, reference, n, tile_size);
    double parallel_seconds =
        benchmark_gemm("gemm_parallel", a, b, parallel, n, tile_size);
    double tiled_seconds =
        benchmark_gemm("gemm_tiled", a, b, tiled, n, tile_size);

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
               : 1;
}

static int run_scaling(size_t dot_n, size_t matrix_n, size_t tile_size)
{
    if (matrix_n > SIZE_MAX / matrix_n ||
        matrix_n * matrix_n > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }

    int logical_processors = omp_get_num_procs();
    if (logical_processors < 1) {
        fprintf(stderr, "OpenMP reported no available logical processors\n");
        return 1;
    }

    int thread_counts[sizeof(int) * CHAR_BIT];
    size_t thread_count = make_thread_counts(
        thread_counts, sizeof(thread_counts) / sizeof(thread_counts[0]),
        logical_processors);
    if (thread_count == 0) {
        fprintf(stderr, "Could not create an OpenMP thread-count sweep\n");
        return 1;
    }

    if (dot_n > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "Dot-product vectors are too large\n");
        return 1;
    }
    double *dot_a = malloc(dot_n * sizeof(*dot_a));
    double *dot_b = malloc(dot_n * sizeof(*dot_b));
    if (dot_a == NULL || dot_b == NULL) {
        fprintf(stderr, "Could not allocate dot-product vectors\n");
        free(dot_a);
        free(dot_b);
        return 1;
    }
    for (size_t i = 0; i < dot_n; ++i) {
        dot_a[i] = 1.0;
        dot_b[i] = 0.5;
    }

    size_t matrix_elements = matrix_n * matrix_n;
    size_t matrix_bytes = matrix_elements * sizeof(double);
    double *a = malloc(matrix_bytes);
    double *b = malloc(matrix_bytes);
    double *c = malloc(matrix_bytes);
    if (a == NULL || b == NULL || c == NULL) {
        fprintf(stderr, "Could not allocate GEMM matrices of size %zu x %zu\n",
                matrix_n, matrix_n);
        free(dot_a);
        free(dot_b);
        free(a);
        free(b);
        free(c);
        return 1;
    }
    for (size_t i = 0; i < matrix_elements; ++i) {
        a[i] = (double)(i % 17) / 17.0;
        b[i] = (double)(i % 19) / 19.0;
    }

    omp_set_dynamic(0);
    printf("A3 thread scaling: %d logical processors available; "
           "%d timed runs after one warm-up\n",
           logical_processors, GEMM_TIMED_RUNS);

    printf("\nReduction dot product (N = %zu)\n", dot_n);
    printf("%-10s %-10s %-14s %-14s %-14s\n", "requested", "actual",
           "time (s)", "speedup", "efficiency");
    double dot_baseline = 0.0;
    for (size_t i = 0; i < thread_count; ++i) {
        int requested_threads = thread_counts[i];
        omp_set_num_threads(requested_threads);
        int actual_threads = actual_openmp_threads();
        double seconds = benchmark_reduction_dot(dot_a, dot_b, dot_n);
        if (seconds < 0.0) {
            free(dot_a);
            free(dot_b);
            free(a);
            free(b);
            free(c);
            return 1;
        }
        if (i == 0) {
            dot_baseline = seconds;
        }
        double speedup = dot_baseline / seconds;
        double efficiency = speedup / (double)actual_threads;
        printf("%-10d %-10d %-14.6f %-14.3f %-14.3f\n",
               requested_threads, actual_threads, seconds, speedup,
               efficiency);
    }

    printf("\nTiled GEMM (N = %zu, tile size = %zu)\n", matrix_n, tile_size);
    printf("%-10s %-10s %-14s %-14s %-14s\n", "requested", "actual",
           "time (s)", "speedup", "efficiency");
    double gemm_baseline = 0.0;
    for (size_t i = 0; i < thread_count; ++i) {
        int requested_threads = thread_counts[i];
        omp_set_num_threads(requested_threads);
        int actual_threads = actual_openmp_threads();
        double seconds =
            benchmark_gemm("gemm_tiled", a, b, c, matrix_n, tile_size);
        if (i == 0) {
            gemm_baseline = seconds;
        }
        double speedup = gemm_baseline / seconds;
        double efficiency = speedup / (double)actual_threads;
        printf("%-10d %-10d %-14.6f %-14.3f %-14.3f\n",
               requested_threads, actual_threads, seconds, speedup,
               efficiency);
    }

    free(dot_a);
    free(dot_b);
    free(a);
    free(b);
    free(c);
    return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
    // scaling
    if (argc >= 2 && strcmp(argv[1], "scaling") == 0) {
        size_t dot_n = DATA_SIZE;
        size_t matrix_n = DEFAULT_GEMM_N;
        size_t tile_size = TILE_SIZE;
        if (argc > 5 ||
            (argc >= 3 && !parse_length(argv[2], &dot_n)) ||
            (argc >= 4 && !parse_length(argv[3], &matrix_n)) ||
            (argc == 5 && !parse_length(argv[4], &tile_size))) {
            fprintf(stderr,
                    "Usage: %s scaling [dot_length [matrix_size [tile_size]]]\n",
                    argv[0]);
            return 1;
        }
        return run_scaling(dot_n, matrix_n, tile_size);
    }

    // GEMM
    printf("GEMM");
    if (argc >= 2 && strcmp(argv[1], "gemm") == 0) {
        size_t n = DEFAULT_GEMM_N;
        size_t tile_size = TILE_SIZE;
        if (argc > 4 ||
            (argc >= 3 && !parse_length(argv[2], &n)) ||
            (argc == 4 && !parse_length(argv[3], &tile_size))) {
            fprintf(stderr,
                    "Usage: %s gemm [matrix_size [tile_size]]\n", argv[0]);
            return 1;
        }
        return run_gemm(n, tile_size);
    }

    // dot product
    {
        size_t n = DATA_SIZE;
        size_t synchronized_n = n;
        if (argc > 3) {
            fprintf(stderr, "Usage: %s [vector_length [critical_atomic_length]]\n", argv[0]);
            return 1;
        }
        if (argc >= 2) {
            if (!parse_length(argv[1], &n)) {
                fprintf(stderr, "vector_length must be a positive valid length\n");
                return 1;
            }
        }
        synchronized_n = n;
        if (argc == 3) {
            if (!parse_length(argv[2], &synchronized_n) || synchronized_n > n) {
                fprintf(stderr,
                        "critical_atomic_length must be positive and no larger "
                        "than vector_length\n");
                return 1;
            }
        }

        double *a = malloc(n * sizeof(*a));
        double *b = malloc(n * sizeof(*b));
        if (a == NULL || b == NULL) {
            fprintf(stderr, "Could not allocate memory for vectors of length %zu\n", n);
            free(a);
            free(b);
            return 1;
        }

        // initialize vectors with known values so that the expected_sum dot product can be compared against the computed result
        for (size_t i = 0; i < n; ++i) {
            a[i] = 1.0;
            b[i] = 0.5;
        }

        const double expected_sum = (double)n / 2.0;
        const double synchronized_expected_sum = (double)synchronized_n / 2.0;
        double sum = 0.0;
        double start = 0.0;
        double time_diff = 0.0;

        printf("Vector length: %zu; critical/atomic length: %zu; "
            "OpenMP max threads: %d\n",
            n, synchronized_n, omp_get_max_threads());

        start = omp_get_wtime();
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            sum += a[i] * b[i];
        }
        time_diff = omp_get_wtime() - start;
        print_result("racy", sum, expected_sum, time_diff);

        sum = 0.0;
        start = omp_get_wtime();
        #pragma omp parallel for
        for (size_t i = 0; i < synchronized_n; ++i) {
            double product = a[i] * b[i];
            #pragma omp critical
            {
                sum += product;
            }
        }
        time_diff = omp_get_wtime() - start;
        print_result("critical", sum, synchronized_expected_sum, time_diff);

        sum = 0.0;
        start = omp_get_wtime();
        #pragma omp parallel for
        for (size_t i = 0; i < synchronized_n; ++i) {
            #pragma omp atomic update
            sum += a[i] * b[i];
        }
        time_diff = omp_get_wtime() - start;
        print_result("atomic", sum, synchronized_expected_sum, time_diff);

        sum = 0.0;
        start = omp_get_wtime();
        #pragma omp parallel for reduction(+ : sum)
        for (size_t i = 0; i < n; ++i) {
            sum += a[i] * b[i];
        }
        time_diff = omp_get_wtime() - start;
        print_result("reduction", sum, expected_sum, time_diff);

        printf("All done!\n");
        free(a);
        free(b);
    }
    return 0;
}
