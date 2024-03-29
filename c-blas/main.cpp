#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <cblas.h>


void print_res_mx(int m, int p, double **res) {
    std::cout << "Resulting matrix is" << std::endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
		std::cout << res[i][j] << ' ';
        }
        std::cout << std::endl;
    }
}

void reset_res_mx(int m, int p, double **res) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            res[i][j] = 0.0;
        }
    }
}

void set_mx(int m, int n, double **mx) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mx[i][j] = (i * j) % 10;
        }
    }
}

void print_C(int M, int N, double *C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", C[i * N + j]);
        }
        printf("\n");
    }
}

void set_MX(int M, int N, double *MX) {
    for (int i = 0; i < (M * N); i++) {
        MX[i] = i % 10;
    }
}

void reset_C(int M, int N, double *C) {
    for (int i = 0; i < (M * N); i++) {
        C[i] = 0.0;
    }
}

void multiply_plain_c(
    const double *A, const double *B, double *C,
    const int M, const int N, const int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

void multiply_avx128(
    const double *A, const double *B, double *C,
    const int M, const int N, const int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __m128d sum = _mm_setzero_pd(); // set sum = [0.0, 0.0]
            for (int k = 0; k < K; k += 2) {
                __m128d a = _mm_load_pd(A + i * K + k); // load two doubles from A into a
                __m128d b = _mm_load_pd(B + k * N + j); // load two doubles from B into b
                __m128d tmp = _mm_dp_pd(a, b, 0x31); // calculate a*b and store it in the lower 64 bits of tmp
                sum = _mm_add_pd(sum, tmp); // add tmp to sum
            }
            _mm_store_sd(C + i * N + j, sum); // store the lower 64 bits of sum to C
        }
    }
}

void multiply_avx256(
    const double *A, const double *B, double *C,
    const int M, const int N, const int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __m256d sum = _mm256_setzero_pd(); // set sum = [0.0, 0.0, 0.0, 0.0]
            for (int k = 0; k < K; k += 4) {
                __m256d a = _mm256_load_pd(A + i * K + k); // load four doubles from A into a
                __m256d b = _mm256_load_pd(B + k * N + j); // load four doubles from B into b
                __m256d tmp = _mm256_mul_pd(a, b); // calculate a*b and store it in tmp
                tmp = _mm256_hadd_pd(tmp, tmp); // horizontally add elements in tmp
                sum = _mm256_add_pd(sum, tmp); // add tmp to sum
            }
            double result[4];
            _mm256_storeu_pd(result, sum); // store the 128 bits of sum to result
            // sum partial results
            C[i * N + j] = result[0] + result[1] + result[2] + result[3];
        }
    }
}

void transpose_matrix(const int K, const int N, const double *A, double *T) {
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            T[k * N + n] = A[n * K + k];
        }
    }
}

void benchmark_matrices(const int times_to_run, const int M, const int N, const int K) {
    std::cout << "Benchmarking for matrices of dimenstions M, N, K: " << M << ", " << N << ", " << K << ", " << std::endl;

    double *A = static_cast<double *>(malloc(sizeof(double) * M * K));
    if (NULL == A) {
        std::cerr << "Could not allocate for A" << std::endl;
        return;
    }
    double *B = static_cast<double *>(malloc(sizeof(double) * K * N));
    if (NULL == B) {
        std::cerr << "Could not allocate for B" << std::endl;
        return;
    }
    double *Bt = static_cast<double *>(malloc(sizeof(double) * K * N));
    if (NULL == Bt) {
        std::cerr << "Could not allocate for Bt" << std::endl;
        return;
    }
    double *C = static_cast<double *>(malloc(sizeof(double) * M * N));
    if (NULL == C) {
        std::cerr << "Could not allocate for C" << std::endl;
        return;
    }
    set_MX(M, K, A);
    set_MX(K, N, B);
    transpose_matrix(K, N, B, Bt);

    std::chrono::time_point<std::chrono::system_clock> start, end;
    long long duration;
    // Plain C

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < times_to_run; i++) {
        reset_C(M, N, C);
        multiply_plain_c(A, B, C, M, N, K);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Plain C took " << duration << std::endl;
    // printf("Plain C result\n");
    // print_res_mx(m, p, res);

    // SIMD
    // start = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < times_to_run; i++) {
    //     reset_C(M, N, C);
    //     multiply_avx128(A, B, C, M, N, K);
    // }
    // end = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // std::cout << "Time for AVX 128bit is " << duration << std::endl;
    //
    // start = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < times_to_run; i++) {
    //     //reset_C(M, N, C);
    //     multiply_avx256(A, B, C, M, N, K);
    // }
    // end = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // std::cout << "Time for AVX 256bit is " << duration << std::endl;


    // OpenBLAS
    reset_C(M, N, C);

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < times_to_run; i++) {
        //reset_C(M, N, C);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 1, C, N);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Time for OpenBLAS is " << duration << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < times_to_run; i++) {
        //reset_C(M, N, C);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1, A, K, Bt, K, 1, C, N);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Time for OpenBLAS transposed is " << duration << std::endl;
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

    int M = 16, N = 16, K = 32;
    benchmark_matrices(times_to_run, M, N, K);

    M = 256, N = 256, K = 512;
    benchmark_matrices(times_to_run, M, N, K);

    M = 1024, N = 1024, K = 2048;
    benchmark_matrices(times_to_run, M, N, K);

    return 0;
}
