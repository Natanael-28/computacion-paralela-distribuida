/*
 * ============================================================
 *  Ejercicio 4 (Reto) - Speedup hibrido
 *  Autores: Josue Natanael Reyes & Marlon Arevalo
 * ============================================================
 */

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk = N / size;
    long long *arr = NULL;

    if (rank == 0) {
        arr = (long long*) malloc(N * sizeof(long long));
        for (int i = 0; i < N; i++) arr[i] = i;
    }

    long long *local = (long long*) malloc(chunk * sizeof(long long));

    MPI_Barrier(MPI_COMM_WORLD);
    double t_inicio = MPI_Wtime();

    MPI_Scatter(arr,   chunk, MPI_LONG_LONG,
                local, chunk, MPI_LONG_LONG,
                0, MPI_COMM_WORLD);

    long long suma_local = 0;
    #pragma omp parallel for reduction(+:suma_local)
    for (int i = 0; i < chunk; i++) {
        suma_local += local[i];
    }

    long long suma_total = 0;
    MPI_Reduce(&suma_local, &suma_total, 1,
               MPI_LONG_LONG, MPI_SUM,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_fin = MPI_Wtime();
    double t_paralelo = t_fin - t_inicio;

    if (rank == 0) {
        printf("============================================\n");
        printf("Configuracion: %d procesos MPI x %d hilos OMP\n",
               size, omp_get_max_threads());
        printf("Suma total       = %lld\n", suma_total);
        printf("Esperado         = %lld\n", (long long)N * (N - 1) / 2);
        printf("Tiempo paralelo  = %.6f s\n", t_paralelo);

        /* ----- Version secuencial ----- */
        long long suma_seq = 0;
        double ts = MPI_Wtime();
        for (int i = 0; i < N; i++) {
            suma_seq += arr[i];
        }
        double te = MPI_Wtime();
        double t_secuencial = te - ts;

        printf("Tiempo secuencial= %.6f s\n", t_secuencial);
        printf("Suma secuencial  = %lld\n", suma_seq);

        if (t_paralelo > 0.0) {
            printf("Speedup          = %.2fx\n",
                   t_secuencial / t_paralelo);
        }
        printf("============================================\n");

        free(arr);
    }

    free(local);
    MPI_Finalize();
    return 0;
}
