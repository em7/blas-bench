package org.example;

import jdk.incubator.vector.*;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * --add-modules jdk.incubator.vector
 * to VM options
 */
public class App {
    private static void initializeMx(double[][] mx) {
        for (int i = 0; i < mx.length; i++) {
            for (int j = 0; j < mx[0].length; j++) {
                mx[i][j] = ((i + 0.1) * (j + 0.1)) % 10;
            }
        }
    }

    private static void clearMx(double[][] mx) {
        for (int i = 0; i < mx.length; i++) {
            for (int j = 0; j < mx[0].length; j++) {
                mx[i][j] = 0.0;
            }
        }
    }

    private static void multiplyMx(double[][] mxA, double[][] mxB, double[][] mxC) {
        for (int i = 0; i < mxA.length; i++) {
            for (int j = 0; j < mxB[0].length; j++) {
                for (int k = 0; k < mxA[0].length; k++) {
                    mxC[i][j] += mxA[i][k] * mxB[k][j];
                }
            }
        }
    }

    private static void multiplyMxSimd(double[][] mxA, double[][] mxBt, double[][] mxC, int m, int n, int k) {
        // B matrix must be transposed !

        final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_MAX;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int p = 0;
                for (; p < SPECIES.loopBound(mxA[i].length); p += SPECIES.length()) {
                    var mask = SPECIES.indexInRange(p, mxA[i].length);
                    var v1 = DoubleVector.fromArray(SPECIES, mxA[i], p, mask);
                    var v2 = DoubleVector.fromArray(SPECIES, mxBt[j], p, mask);
                    mxC[i][j] += v1.mul(v2).reduceLanes(VectorOperators.ADD);
                }

                for (; p< mxA[i].length; p++) {
                    mxC[i][j] += mxA[i][p] * mxBt[j][p];
                }
            }
        }
    }

    private static double[][] transposeMatrix(double[][] m) {
        double[][] temp = new double[m[0].length][m.length];
        for (int i = 0; i < m.length; i++)
            for (int j = 0; j < m[0].length; j++)
                temp[j][i] = m[i][j];
        return temp;
    }

    private static void benchmark_matrices(int m, int n, int k, int iterations) {
        double[][] mxA = new double[m][k];
        double[][] mxB = new double[k][n];
        double[][] mxBt;
        double[][] mxC = new double[m][n];
        long startNanoSec, endNanoSec, durationUs;

        String backend = Nd4j.getBackend().getClass().getSimpleName();
        System.out.println(backend);

        initializeMx(mxA);
        initializeMx(mxB);
        mxBt = transposeMatrix(mxB);
        clearMx(mxC);

        System.out.printf("Benchmarking matrices m, n, k: %d, %d, %d\n", m, n, k);

        System.out.println("Starting Plain Java");
        startNanoSec = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            multiplyMx(mxA, mxB, mxC);
        }
        endNanoSec = System.nanoTime();
        durationUs = (endNanoSec - startNanoSec) / 1000;
        System.out.println("Plain Java took " + durationUs + " us");

        INDArray ndA = Nd4j.create(mxA);
        INDArray ndB = Nd4j.create(mxB);
        INDArray ndC = Nd4j.create(mxC);

        System.out.println("Starting SIMD max");
        startNanoSec = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            multiplyMxSimd(mxA, mxBt, mxC, m, n, k);
        }
        endNanoSec = System.nanoTime();
        durationUs = (endNanoSec - startNanoSec) / 1000;
        System.out.println("SIMD max took " + durationUs + " us");

        System.out.println("Starting ND4j");
        startNanoSec = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            Nd4j.matmul(ndA, ndB, ndC);
        }
        endNanoSec = System.nanoTime();
        durationUs = (endNanoSec - startNanoSec) / 1000;
        System.out.println("ND4j took " + durationUs + " us");
    }

    public static void main(String[] args) {
        System.out.println("First, warmup");

        int iterations = 1;
        benchmark_matrices(16, 16, 32, 100);

        System.out.println("\n\nNow for real:");

        benchmark_matrices(16, 16, 32, iterations);
        benchmark_matrices(256, 256, 512, iterations);
        benchmark_matrices(1024, 1024, 2048, iterations);
    }
}
