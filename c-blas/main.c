#include <stdio.h>
#include <sys/time.h>
#include <immintrin.h>
#include <cblas.h>

void print_res_mx(const int m, const int p, double res[m][p]) {
    printf("Resulting matrix is\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            printf("%5.2f ", res[i][j]);
        }
        printf("\n");
    }
}

void reset_res_mx(const int m, const int p, double res[m][p]) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            res[i][j] = 0.0;
        }
    }
}

void set_mx(const int m, const int n, double mx[m][n]) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mx[i][j] = (i * j) % 10;
        }
    }
}

void print_C(const int M, const int N, double* C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", C[i*N + j]);
        }
        printf("\n");
    }
}

void set_MX(const int M, const int N, double* MX) {
    for (int i = 0; i < (M * N); i++) {
        MX[i] = i % 10;
    }
}

void reset_C(const int M, const int N, double* C) {
    for (int i = 0; i < (M * N); i++) {
        C[i] = 0.0;
    }
}

void multiply_plain_c(
    const double* A, const double* B, double* C,
    const int M, const int N, const int K) {

    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            for(int k = 0; k < K; k++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

void multiply_avx128(
    const double* A, const double* B, double* C,
    const int M, const int N, const int K) {

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __m128d sum = _mm_setzero_pd();  // set sum = [0.0, 0.0]
            for (int k = 0; k < K; k += 2) {
                __m128d a = _mm_load_pd(A + i*K + k);  // load two doubles from A into a
                __m128d b = _mm_load_pd(B + k*N + j);  // load two doubles from B into b
                __m128d tmp = _mm_dp_pd(a, b, 0x31);   // calculate a*b and store it in the lower 64 bits of tmp
                sum = _mm_add_pd(sum, tmp);   // add tmp to sum
            }
            _mm_store_sd(C + i*N + j, sum);  // store the lower 64 bits of sum to C
        }
    }
}

void multiply_avx256(
    const double* A, const double* B, double* C,
    const int M, const int N, const int K) {

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __m256d sum = _mm256_setzero_pd();  // set sum = [0.0, 0.0, 0.0, 0.0]
            for (int k = 0; k < K; k += 4) {
                __m256d a = _mm256_load_pd(A + i*K + k);  // load four doubles from A into a
                __m256d b = _mm256_load_pd(B + k*N + j);  // load four doubles from B into b
                __m256d tmp = _mm256_mul_pd(a, b);   // calculate a*b and store it in tmp
                tmp = _mm256_hadd_pd(tmp, tmp); // horizontally add elements in tmp
                sum = _mm256_add_pd(sum, tmp);   // add tmp to sum
            }
            double result[4];
            _mm256_storeu_pd(result, sum); // store the 128 bits of sum to result
            // sum partial results
            C[i*N + j] = result[0]+result[1]+result[2]+result[3];
        }
    }
}

void multiply_openBLAS(double* A, double* B, double* C, const int M, const int K, const int N) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, M, B, K, 1, C, M);
}


void benchmark_matrices(const int times_to_run, const int M, const int N, const int K) {
    printf("Benchmarking for matrices of dimenstions M, N, K: %d, %d, %d\n", M, N, K);

    struct timeval start, end;
    double *A = malloc(sizeof(double)*M*K);
    if (NULL == A) {
        fprintf(stderr, "Could not allocate for A\n");
        return;
    }
    double *B = malloc(sizeof(double)*K*N);
    if (NULL == B) {
        fprintf(stderr, "Could not allocate for B\n");
        return;
    }
    double *C = malloc(sizeof(double)*M*N);
    if (NULL == C) {
        fprintf(stderr, "Could not allocate for C\n");
        return;
    }
    set_MX(M, K, A);
    set_MX(K, N, B);

    // Plain C
    gettimeofday(&start, NULL);
    for (int i = 0; i < times_to_run; i++) {
        reset_C(M, N, C);
        multiply_plain_c(A, B, C, M, N, K);
    }
    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Time for Plain C is %ld us\n", micros);
    // printf("Plain C result\n");
    // print_res_mx(m, p, res);

    // SIMD
    gettimeofday(&start, NULL);
    for (int i = 0; i < times_to_run; i++) {
        reset_C(M, N, C);
        multiply_avx128(A, B, C, M, N, K);
    }
    gettimeofday(&end, NULL);
    seconds = (end.tv_sec - start.tv_sec);
    micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Time for AVX 128bit is %ld us\n", micros);

    gettimeofday(&start, NULL);
    for (int i = 0; i < times_to_run; i++) {
        //reset_C(M, N, C);
        multiply_avx256(A, B, C, M, N, K);
    }
    gettimeofday(&end, NULL);
    seconds = (end.tv_sec - start.tv_sec);
    micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Time for AVX 256bit %ld us\n", micros);


    // OpenBLAS
    reset_C(M, N, C);

    gettimeofday(&start, NULL);
    for (int i = 0; i < times_to_run; i++) {
        //reset_C(M, N, C);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 1, C, N);
    }
    gettimeofday(&end, NULL);
    seconds = (end.tv_sec - start.tv_sec);
    micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Time for OpenBLAS is %ld us\n", micros);
    // printf("OpenBLAS result\n");
    // print_C(M, N, C);

    free(A);
    free(B);
    free(C);
}

int main() {
    printf("Runnin Plain C!\n");

    //const int times_to_run = 100;
    const int times_to_run = 1;

    int M = 15, N = 15, K = 30;
    benchmark_matrices(times_to_run, M, N, K);

    M = 150, N = 150, K = 300;
    benchmark_matrices(times_to_run, M, N, K);

    M = 1500, N = 1500, K = 3000;
    benchmark_matrices(times_to_run, M, N, K);

    return 0;
}
