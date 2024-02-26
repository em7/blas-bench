import numpy as np
import time

def multiply_plain_python(A, B, m, k, n):
    # Create the result matrix C
    # Dimensions would be m x n
    C = [[0 for j in range(n)] for i in range(m)]

    # Implementing the formula of matrix multiplication
    for i in range(m):
        for j in range(n):
            for l in range(k):
                C[i][j] += A[i][l] * B[l][j]

    return C

def multiply_numpy(A, B):
    return np.matmul(A, B)


def benchmark(m, n, k, number_of_iterations):
    print(f"Benchmarking with dimensions m, n, k: {m}, {n}, {k}")
    A_python = [[((i * j) % 10) + 0.1 for j in range(k)] for i in range(m)]
    B_python = [[((i * j) % 10) + 0.1 for j in range(n)] for i in range(k)]

    A = np.array(A_python)
    B = np.array(B_python)
    print("Starting NumPy")
    start = time.perf_counter_ns()
    for it in range(number_of_iterations):
        C_numpy = multiply_numpy(A, B)
    end = time.perf_counter_ns()
    elapsed = (end - start) / 1000
    print(f"Numpy took {elapsed} us.")

    print("Starting plain Python")
    start = time.perf_counter_ns()
    for it in range(number_of_iterations):
        C_python = multiply_plain_python(A_python, B_python, m, k, n)
    end = time.perf_counter_ns()
    elapsed = (end - start) / 1000
    print(f"Plain Python took {elapsed} us.")




number_of_iterations = 1
benchmark(16, 16, 32, number_of_iterations)

benchmark(256, 256, 512, number_of_iterations)

benchmark(1024, 1024, 2048, number_of_iterations)


