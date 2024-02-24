package org.example;

//import jdk.incubator.vector.*;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * --add-modules jdk.incubator.vector
 * to VM options
 *
 */
public class App 
{
    private static void initialize_mx(double[][] mx) {
        for (int i = 0; i < mx.length; i++) {
            for (int j = 0; j < mx[0].length; j++) {
                mx[i][j] = (i * j) % 10;
            }
        }
    }

    private static void clear_mx(double[][] mx) {
        for (int i = 0; i < mx.length; i++) {
            for (int j = 0; j < mx[0].length; j++) {
                mx[i][j] = 0.0;
            }
        }
    }

    private static void multiply_mx(double[][] mxA, double[][] mxB, double[][] mxC){
        for (int i = 0; i < mxA.length; i++) {
            for (int j = 0; j < mxB[0].length; j++) {
                for (int k = 0; k < mxA[0].length; k++) {
                    mxC[i][j] += mxA[i][k] * mxB[k][j];
                }
            }
        }
    }

//    private static void multiply_mx_simd(double[][] mxA, double[][] mxB, double[][] mxC, int m, int n, int k) {
//        final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_MAX;
//
//        for (int i = 0; i < m; i++) {
//            for (int j = 0; j < n; j++) {
//                DoubleVector c = DoubleVector.zero(SPECIES);
//                int p = 0;
//                double[] mA = mxA[i];
//                for (; p <= k - SPECIES.length(); p += SPECIES.length()) {
//                    DoubleVector a = DoubleVector.fromArray(SPECIES, mA, p);
//                    double[] columnB = new double[SPECIES.length()];
//                    for (int index = 0; index < SPECIES.length(); index++) {
//                        columnB[index] = mxB[p + index][j];
//                    }
//                    DoubleVector b = DoubleVector.fromArray(SPECIES, columnB, 0);
//                    c = a.fma(b, c);  //performs a fused multiply-add (a * b + c)
//                }
//
//                double sum = 0.0;
//                // Now handle the tail part (remainder) of the array
//                for(; p < k; p++) {
//                    sum += mxA[i][p] * mxB[p][j];
//                }
//
//                mxC[i][j] = c.reduceLanes(VectorOperators.ADD) + sum;
//            }
//        }
//    }

    private static void benchmark_matrices(int m, int n, int k, int iterations) {
        double[][] mxA = new double[m][k];
        double[][] mxB = new double[k][n];
        double[][] mxC = new double[m][n];
        long startNanoSec, endNanoSec, durationUs;

        initialize_mx(mxA);
        initialize_mx(mxB);
        clear_mx(mxC);

        System.out.printf("Benchmarking matrices m, n, k: %d, %d, %d\n", m, n, k);

        System.out.println("Starting Plain Java");
        startNanoSec = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            multiply_mx(mxA, mxB, mxC);
        }
        endNanoSec = System.nanoTime();
        durationUs = (endNanoSec - startNanoSec) / 1000;
        System.out.println("Plain Java took " + durationUs + "us");

        INDArray ndA = Nd4j.create(mxA);
        INDArray ndB = Nd4j.create(mxB);
        INDArray ndC = Nd4j.create(mxC);

//        System.out.println("Starting SIMD max");
//        startNanoSec = System.nanoTime();
//        for (int i = 0; i < iterations; i++) {
//            multiply_mx_simd(mxA, mxB, mxC, m, n, k);
//        }
//        endNanoSec = System.nanoTime();
//        durationUs = (endNanoSec - startNanoSec) / 1000;
//        System.out.println("SIMD max took " + durationUs + "us");

        System.out.println("Starting ND4j");
        startNanoSec = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            Nd4j.matmul(ndA, ndB, ndC);
        }
        endNanoSec = System.nanoTime();
        durationUs = (endNanoSec - startNanoSec) / 1000;
        System.out.println("ND4j took " + durationUs + "us");
    }

    public static void main( String[] args )
    {
        System.out.println( "First, warmup" );

        int iterations = 1;
        benchmark_matrices(16, 16, 32, 100);

        System.out.println( "\n\nNow for real:" );

        benchmark_matrices(16, 16, 32, iterations);
        benchmark_matrices(256, 256, 512, iterations);
        benchmark_matrices(1024, 1024, 2048, iterations);
    }
}
