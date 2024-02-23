package org.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void initialize_mx(double[][] mx) {
        for (int i = 0; i < mx.length; i++) {
            for (int j = 0; j < mx[0].length; j++) {
                mx[i][j] = (i * j) % 10;
            }
        }
    }

    public static void clear_mx(double[][] mx) {
        for (int i = 0; i < mx.length; i++) {
            for (int j = 0; j < mx[0].length; j++) {
                mx[i][j] = 0.0;
            }
        }
    }

    public static void multiply_mx(double[][] mxA, double[][] mxB, double[][] mxC){
        for (int i = 0; i < mxA.length; i++) {
            for (int j = 0; j < mxB[0].length; j++) {
                for (int k = 0; k < mxA[0].length; k++) {
                    mxC[i][j] += mxA[i][k] * mxB[k][j];
                }
            }
        }
    }

    public static void main( String[] args )
    {
        System.out.println( "First, warmup" );

        int iterations = 1;
        benchmark_matrices(15, 15, 30, 100);

        System.out.println( "\n\nNow for real:" );

        benchmark_matrices(15, 15, 30, iterations);
        benchmark_matrices(150, 150, 300, iterations);
        benchmark_matrices(1500, 1500, 3000, iterations);
    }

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

        System.out.println("Starting ND4j");
        startNanoSec = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            Nd4j.matmul(ndA, ndB, ndC);
        }
        endNanoSec = System.nanoTime();
        durationUs = (endNanoSec - startNanoSec) / 1000;
        System.out.println("ND4j took " + durationUs + "us");
    }
}
