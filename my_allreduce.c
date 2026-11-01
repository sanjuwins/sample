#include <errno.h>
#include <limits.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_VECTOR_LENGTH 1003
#define ERROR_TOLERANCE 1.0e-10

static const size_t benchmark_lengths[] = {
    (size_t)1 << 4,  (size_t)1 << 8,  (size_t)1 << 12,
    (size_t)1 << 16, (size_t)1 << 20, (size_t)1 << 24};

static void chunk_bounds(size_t n, int process_count, int chunk,
                         size_t *start, size_t *count)
{
    size_t processes = (size_t)process_count;
    size_t chunk_index = (size_t)chunk;
    size_t base = n / processes;
    size_t remainder = n % processes;

    *start = base * chunk_index +
             (size_t)(((uint64_t)remainder * (uint64_t)chunk_index) /
                      (uint64_t)process_count);

    size_t next_start =
        base * (chunk_index + 1) +
        (size_t)(((uint64_t)remainder * (uint64_t)(chunk_index + 1)) /
                 (uint64_t)process_count);
    *count = next_start - *start;
}

int ring_allreduce(double *buf, size_t n, MPI_Comm comm)
{
    int rank = 0;
    int process_count = 0;
    int mpi_result = MPI_Comm_rank(comm, &rank);
    if (mpi_result != MPI_SUCCESS) {
        return mpi_result;
    }
    mpi_result = MPI_Comm_size(comm, &process_count);
    if (mpi_result != MPI_SUCCESS) {
        return mpi_result;
    }
    if (n > 0 && buf == NULL) {
        return MPI_ERR_BUFFER;
    }
    if (process_count <= 0) {
        return MPI_ERR_COMM;
    }

    size_t max_chunk_count =
        n / (size_t)process_count +
        (n % (size_t)process_count != 0 ? 1U : 0U);
    if (max_chunk_count > (size_t)INT_MAX ||
        max_chunk_count > SIZE_MAX / sizeof(double)) {
        return MPI_ERR_COUNT;
    }

    double *incoming = NULL;
    if (max_chunk_count > 0) {
        incoming = malloc(max_chunk_count * sizeof(*incoming));
        if (incoming == NULL) {
            return MPI_ERR_NO_MEM;
        }
    }

    int next_rank = (rank + 1) % process_count;
    int previous_rank = (rank - 1 + process_count) % process_count;

    for (int step = 0; step < process_count - 1; ++step) {
        int send_chunk = (rank - step + process_count) % process_count;
        int receive_chunk =
            (rank - step - 1 + process_count) % process_count;
        size_t send_start;
        size_t send_count;
        size_t receive_start;
        size_t receive_count;
        chunk_bounds(n, process_count, send_chunk, &send_start, &send_count);
        chunk_bounds(n, process_count, receive_chunk, &receive_start,
                     &receive_count);

        mpi_result = MPI_Sendrecv(
            send_count > 0 ? &buf[send_start] : buf, (int)send_count,
            MPI_DOUBLE, next_rank, 0,
            receive_count > 0 ? incoming : buf, (int)receive_count,
            MPI_DOUBLE, previous_rank, 0, comm, MPI_STATUS_IGNORE);
        if (mpi_result != MPI_SUCCESS) {
            free(incoming);
            return mpi_result;
        }

        for (size_t i = 0; i < receive_count; ++i) {
            buf[receive_start + i] += incoming[i];
        }
    }

    for (int step = 0; step < process_count - 1; ++step) {
        int send_chunk = (rank - step + 1 + process_count) % process_count;
        int receive_chunk = (rank - step + process_count) % process_count;
        size_t send_start;
        size_t send_count;
        size_t receive_start;
        size_t receive_count;
        chunk_bounds(n, process_count, send_chunk, &send_start, &send_count);
        chunk_bounds(n, process_count, receive_chunk, &receive_start,
                     &receive_count);

        mpi_result = MPI_Sendrecv(
            send_count > 0 ? &buf[send_start] : buf, (int)send_count,
            MPI_DOUBLE, next_rank, 1,
            receive_count > 0 ? &buf[receive_start] : buf,
            (int)receive_count, MPI_DOUBLE, previous_rank, 1, comm,
            MPI_STATUS_IGNORE);
        if (mpi_result != MPI_SUCCESS) {
            free(incoming);
            return mpi_result;
        }
    }

    free(incoming);
    return MPI_SUCCESS;
}

static int benchmark_iterations(size_t n)
{
    if (n <= ((size_t)1 << 8)) {
        return 1000;
    }
    if (n <= ((size_t)1 << 12)) {
        return 500;
    }
    if (n <= ((size_t)1 << 16)) {
        return 100;
    }
    if (n <= ((size_t)1 << 20)) {
        return 10;
    }
    return 3;
}

static int time_ring_allreduce(const double *input, double *work, size_t n,
                               size_t bytes, int iterations, MPI_Comm comm,
                               double *max_seconds)
{
    int mpi_result;
    memcpy(work, input, bytes);
    mpi_result = ring_allreduce(work, n, comm);
    if (mpi_result != MPI_SUCCESS) {
        return mpi_result;
    }

    double elapsed_seconds = 0.0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        memcpy(work, input, bytes);
        mpi_result = MPI_Barrier(comm);
        if (mpi_result != MPI_SUCCESS) {
            return mpi_result;
        }

        double start = MPI_Wtime();
        mpi_result = ring_allreduce(work, n, comm);
        double end = MPI_Wtime();
        if (mpi_result != MPI_SUCCESS) {
            return mpi_result;
        }
        elapsed_seconds += end - start;
    }

    double local_average = elapsed_seconds / (double)iterations;
    return MPI_Reduce(&local_average, max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                      comm);
}

static int time_mpi_allreduce(const double *input, double *work, size_t n,
                              size_t bytes, int iterations, MPI_Comm comm,
                              double *max_seconds)
{
    int mpi_result;
    memcpy(work, input, bytes);
    mpi_result =
        MPI_Allreduce(MPI_IN_PLACE, work, (int)n, MPI_DOUBLE, MPI_SUM, comm);
    if (mpi_result != MPI_SUCCESS) {
        return mpi_result;
    }

    double elapsed_seconds = 0.0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        memcpy(work, input, bytes);
        mpi_result = MPI_Barrier(comm);
        if (mpi_result != MPI_SUCCESS) {
            return mpi_result;
        }

        double start = MPI_Wtime();
        mpi_result =
            MPI_Allreduce(MPI_IN_PLACE, work, (int)n, MPI_DOUBLE, MPI_SUM,
                          comm);
        double end = MPI_Wtime();
        if (mpi_result != MPI_SUCCESS) {
            return mpi_result;
        }
        elapsed_seconds += end - start;
    }

    double local_average = elapsed_seconds / (double)iterations;
    return MPI_Reduce(&local_average, max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                      comm);
}

static int run_b2_benchmark(int rank, int process_count, MPI_Comm comm)
{
    if (process_count < 2) {
        if (rank == 0) {
            fprintf(stderr, "B2 requires at least 2 MPI ranks\n");
        }
        return MPI_ERR_COMM;
    }

    double first_ring_time = 0.0;
    double largest_ring_time = 0.0;
    size_t largest_n = benchmark_lengths[
        sizeof(benchmark_lengths) / sizeof(benchmark_lengths[0]) - 1];
    size_t largest_bytes = largest_n * sizeof(double);

    if (rank == 0) {
        printf("B2 all-reduce timing with %d ranks\n", process_count);
        printf("%-12s %-12s %-18s %-18s\n", "elements", "iterations",
               "ring max avg (s)", "MPI max avg (s)");
    }

    for (size_t index = 0;
         index < sizeof(benchmark_lengths) / sizeof(benchmark_lengths[0]);
         ++index) {
        size_t n = benchmark_lengths[index];
        size_t bytes = n * sizeof(double);
        int iterations = benchmark_iterations(n);
        double *input = malloc(bytes);
        double *work = malloc(bytes);
        int allocation_failed = input == NULL || work == NULL;
        int any_allocation_failed = 0;
        int mpi_result =
            MPI_Allreduce(&allocation_failed, &any_allocation_failed, 1,
                          MPI_INT, MPI_MAX, comm);
        if (mpi_result != MPI_SUCCESS) {
            free(input);
            free(work);
            return mpi_result;
        }
        if (any_allocation_failed) {
            if (rank == 0) {
                fprintf(stderr,
                        "Could not allocate B2 vectors for %zu elements\n", n);
            }
            free(input);
            free(work);
            return MPI_ERR_NO_MEM;
        }

        for (size_t i = 0; i < n; ++i) {
            input[i] = (double)(rank + 1) / (double)(i + 1);
        }

        double ring_max_seconds = 0.0;
        double mpi_max_seconds = 0.0;
        mpi_result = time_ring_allreduce(input, work, n, bytes, iterations,
                                         comm, &ring_max_seconds);
        if (mpi_result == MPI_SUCCESS) {
            mpi_result = time_mpi_allreduce(input, work, n, bytes, iterations,
                                            comm, &mpi_max_seconds);
        }
        free(input);
        free(work);
        if (mpi_result != MPI_SUCCESS) {
            return mpi_result;
        }

        if (rank == 0) {
            printf("%-12zu %-12d %-18.9g %-18.9g\n", n, iterations,
                   ring_max_seconds, mpi_max_seconds);
            if (index == 0) {
                first_ring_time = ring_max_seconds;
            }
            if (n == largest_n) {
                largest_ring_time = ring_max_seconds;
            }
        }
    }

    if (rank == 0) {
        double latency_seconds =
            first_ring_time / (2.0 * (double)(process_count - 1));
        double bandwidth_factor =
            2.0 * (double)(process_count - 1) / (double)process_count;
        double largest_latency =
            2.0 * (double)(process_count - 1) * latency_seconds;
        double beta_seconds_per_byte =
            (largest_ring_time - largest_latency) /
            (bandwidth_factor * (double)largest_bytes);

        printf("\nApproximate ring-model estimates "
               "(T = 2(P-1)alpha + 2(P-1)/P * bytes * beta):\n");
        printf("alpha: %.6g us per message step\n",
               latency_seconds * 1.0e6);
        if (beta_seconds_per_byte > 0.0) {
            double effective_bandwidth_gb_per_second =
                1.0 / beta_seconds_per_byte / 1.0e9;
            double crossover_bytes =
                (double)process_count * latency_seconds /
                beta_seconds_per_byte;
            printf("beta: %.6g s/byte; effective bandwidth: %.6g GB/s\n",
                   beta_seconds_per_byte,
                   effective_bandwidth_gb_per_second);
            printf("latency/bandwidth crossover: %.6g bytes "
                   "(%.6g doubles)\n",
                   crossover_bytes, crossover_bytes / sizeof(double));
        } else {
            fprintf(stderr,
                    "Could not estimate positive beta: largest-message "
                    "timing did not exceed the latency estimate\n");
        }
    }

    return MPI_SUCCESS;
}

static int parse_vector_length(const char *text, size_t *n)
{
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno != 0 || text[0] == '-' || end == text || *end != '\0' ||
        parsed == 0 || parsed > (unsigned long long)INT_MAX ||
        parsed > (unsigned long long)(SIZE_MAX / sizeof(double))) {
        return 0;
    }
    *n = (size_t)parsed;
    return 1;
}

static int check_ring_allreduce(size_t n, int rank, int process_count,
                                MPI_Comm comm)
{
    double *values = malloc(n * sizeof(*values));
    double *expected = malloc(n * sizeof(*expected));
    int allocation_failed = values == NULL || expected == NULL;
    int any_allocation_failed = 0;
    int mpi_result = MPI_Allreduce(&allocation_failed, &any_allocation_failed,
                                   1, MPI_INT, MPI_MAX, comm);
    if (mpi_result != MPI_SUCCESS) {
        free(values);
        free(expected);
        return mpi_result;
    }
    if (any_allocation_failed) {
        if (rank == 0) {
            fprintf(stderr, "Could not allocate test vectors\n");
        }
        free(values);
        free(expected);
        return MPI_ERR_NO_MEM;
    }

    srand(1234U + (unsigned int)rank);
    for (size_t i = 0; i < n; ++i) {
        values[i] = (double)rand() / (double)RAND_MAX - 0.5;
    }
    for (size_t i = 0; i < n; ++i) {
        expected[i] = values[i];
    }

    mpi_result = ring_allreduce(values, n, comm);
    if (mpi_result != MPI_SUCCESS) {
        free(values);
        free(expected);
        return mpi_result;
    }

    mpi_result = MPI_Allreduce(MPI_IN_PLACE, expected, (int)n, MPI_DOUBLE,
                               MPI_SUM, comm);
    if (mpi_result != MPI_SUCCESS) {
        free(values);
        free(expected);
        return mpi_result;
    }

    double local_max_difference = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double difference = values[i] - expected[i];
        if (difference < 0.0) {
            difference = -difference;
        }
        if (difference > local_max_difference) {
            local_max_difference = difference;
        }
    }

    double global_max_difference = 0.0;
    mpi_result = MPI_Allreduce(&local_max_difference, &global_max_difference,
                               1, MPI_DOUBLE, MPI_MAX, comm);
    if (mpi_result == MPI_SUCCESS && rank == 0) {
        printf("Ranks: %d, vector length: %zu, max absolute difference: %.17g "
               "(%s; tolerance %.1e)\n",
               process_count, n, global_max_difference,
               global_max_difference < ERROR_TOLERANCE ? "PASS" : "FAIL",
               ERROR_TOLERANCE);
    }

    free(values);
    free(expected);
    if (mpi_result != MPI_SUCCESS) {
        return mpi_result;
    }
    return global_max_difference < ERROR_TOLERANCE ? MPI_SUCCESS
                                                   : MPI_ERR_OTHER;
}

int main(int argc, char **argv)
{
    int mpi_result = MPI_Init(&argc, &argv);
    if (mpi_result != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init failed\n");
        return EXIT_FAILURE;
    }

    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    int rank = 0;
    int process_count = 0;
    mpi_result = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (mpi_result == MPI_SUCCESS) {
        mpi_result = MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    }

    if (mpi_result == MPI_SUCCESS && argc >= 2 &&
        strcmp(argv[1], "b2") == 0) {
        if (argc != 2) {
            if (rank == 0) {
                fprintf(stderr, "Usage: mpiexec -n <ranks> %s b2\n", argv[0]);
            }
            MPI_Finalize();
            return EXIT_FAILURE;
        }
        mpi_result = run_b2_benchmark(rank, process_count, MPI_COMM_WORLD);
        if (mpi_result != MPI_SUCCESS && rank == 0) {
            fprintf(stderr, "B2 benchmark failed (MPI error %d)\n",
                    mpi_result);
        }
        int finalize_result = MPI_Finalize();
        return mpi_result == MPI_SUCCESS && finalize_result == MPI_SUCCESS
                   ? EXIT_SUCCESS
                   : EXIT_FAILURE;
    }

    size_t n = DEFAULT_VECTOR_LENGTH;
    int arguments_valid = argc <= 2;
    if (arguments_valid && argc == 2) {
        arguments_valid = parse_vector_length(argv[1], &n);
    }

    int any_invalid_arguments = 0;
    int local_invalid_arguments = !arguments_valid;
    if (mpi_result == MPI_SUCCESS) {
        mpi_result = MPI_Allreduce(&local_invalid_arguments,
                                   &any_invalid_arguments, 1, MPI_INT,
                                   MPI_MAX, MPI_COMM_WORLD);
    }
    if (mpi_result == MPI_SUCCESS && any_invalid_arguments) {
        if (rank == 0) {
            fprintf(stderr,
                    "Usage: mpiexec -n <ranks> %s [vector_length | b2]\n",
                    argv[0]);
            fprintf(stderr,
                    "vector_length must be a positive integer no larger "
                    "than %d\n",
                    INT_MAX);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (mpi_result == MPI_SUCCESS) {
        mpi_result = check_ring_allreduce(n, rank, process_count,
                                          MPI_COMM_WORLD);
    }

    if (mpi_result != MPI_SUCCESS && rank == 0) {
        fprintf(stderr, "B1 correctness test failed (MPI error %d)\n",
                mpi_result);
    }

    int finalize_result = MPI_Finalize();
    if (mpi_result != MPI_SUCCESS || finalize_result != MPI_SUCCESS) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
