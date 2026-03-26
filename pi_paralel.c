#include <stdio.h>
#include <omp.h>

// Cantidad máxima de hilos permitida por este programa.
// Se usa para dimensionar el arreglo 'sum', donde cada hilo
// acumula su suma parcial en una posición distinta.
#define MAX_THREADS 100

// Número de configuraciones de hilos que vamos a probar.
// En este caso: 10, 20, 30, ... , 100.
#define NUM_CASOS 10

// 'paso' representa Δx en la fórmula de Riemann:
// pi ≈ Δx * Σ f(x_i), con f(x) = 4/(1 + x^2).
double paso;

int main() {
    // Número total de subintervalos/pasos de integración.
    // Mientras más grande, mayor precisión (y más tiempo de ejecución).
    long long num_pasos = 100000000000;

    // Lista de hilos que se probarán para comparar rendimiento.
    // Cada valor se usará en una corrida completa del cálculo de pi.
    int threads_prueba[NUM_CASOS] = {
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100
    };

    // Encabezado CSV para exportar resultados y graficar después:
    // pasos, número de hilos, valor de pi aproximado, tiempo en segundos.
    printf("pasos,threads,pi,tiempo\n");

    // Recorremos cada caso de prueba (cada cantidad de hilos).
    for (int caso = 0; caso < NUM_CASOS; caso++) {
        // Número de hilos para esta ejecución.
        int num_threads = threads_prueba[caso];

        // 'nthreads' guardará cuántos hilos reales creó OpenMP.
        // 'pi' almacenará el resultado final de la aproximación.
        int i, nthreads;
        double pi, sum[MAX_THREADS];

        // Δx = 1 / num_pasos, porque el intervalo es [0, 1].
        paso = 1.0 / (double) num_pasos;

        // Le pedimos a OpenMP usar esta cantidad de hilos.
        omp_set_num_threads(num_threads);

        // Tiempo de inicio de la corrida actual.
        const double ST = omp_get_wtime();

        // Región paralela: todos los hilos ejecutan este bloque.
        #pragma omp parallel
        {
            // 'i' en 64 bits para soportar num_pasos muy grande.
            long long i;
            // 'id' = índice del hilo actual.
            // 'nthrds' = total de hilos activos en la región paralela.
            int id, nthrds;
            // Punto x donde evaluamos la función en cada iteración.
            double x;

            id = omp_get_thread_num();
            nthrds = omp_get_num_threads();

            // Guardamos el número real de hilos (solo lo hace el hilo 0).
            if (id == 0) nthreads = nthrds;

            // Distribución manual del trabajo entre hilos:
            // cada hilo empieza en su 'id' y avanza en saltos de 'nthrds'.
            // Así se reparten las iteraciones sin traslape.
            // sum[id] guarda la suma parcial del hilo 'id'.
            for (i = id, sum[id] = 0.0; i < num_pasos; i = i + nthrds) {
                // Punto medio del subintervalo i:
                // x_i = (i + 0.5) * Δx
                x = (i + 0.5) * paso;

                // Acumulamos f(x_i) = 4/(1+x_i^2) en la suma parcial del hilo.
                sum[id] += 4.0 / (1.0 + x * x);
            }
        }

        // Combinamos las sumas parciales de todos los hilos:
        // pi ≈ Δx * (sum[0] + sum[1] + ... + sum[nthreads-1]).
        for (i = 0, pi = 0.0; i < nthreads; i++) {
            pi += sum[i] * paso;
        }

        // Tiempo de fin de esta corrida.
        const double STOP = omp_get_wtime();

        // Resultado de la corrida actual en formato CSV.
        printf("%lld,%d,%.15f,%f\n", num_pasos, num_threads, pi, (STOP - ST));
    }

    return 0;
}