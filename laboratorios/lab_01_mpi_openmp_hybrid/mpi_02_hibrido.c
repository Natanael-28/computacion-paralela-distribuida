/*
 * ============================================================
 *  Ejercicio 2 - OpenMP dentro de MPI (modelo hibrido)
 *  Autores: Josue Natanael Reyes & Marlon Arevalo
 * ============================================================
 */

#include <mpi.h>
#include <omp.h>
#include <stdio.h>

int main(int argc, char** argv) {
    
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

   
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();        /* ID del hilo (0..3) */
        int nthreads = omp_get_num_threads();  /* Total de hilos en la region */
        printf("  Proceso MPI %d │ Hilo OpenMP %d de %d\n",
               rank, tid, nthreads);
    }

    if (rank == 0) {
        int total_hilos_por_proceso = 4;
        printf("Total unidades: %d x %d = %d\n",
               size,
               total_hilos_por_proceso,
               size * total_hilos_por_proceso);
    }

    MPI_Finalize();
    return 0;
}
