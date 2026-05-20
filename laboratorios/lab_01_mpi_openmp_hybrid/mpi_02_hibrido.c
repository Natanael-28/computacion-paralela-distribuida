/*
 * ============================================================
 *  Ejercicio 2 - OpenMP dentro de MPI (modelo hibrido)
 *  Lab 01 - Programacion Hibrida MPI + OpenMP
 *  Autores: Josue Natanael Reyes & Marlon Arevalo
 * ============================================================
 *
 *  Cada proceso MPI lanza una region paralela OpenMP con
 *  4 hilos. Cada hilo imprime su ID junto con el rank del
 *  proceso al que pertenece. El maestro imprime el total de
 *  unidades de computo (procesos x hilos).
 *
 *  Compilacion:  mpicc -fopenmp mpi_02_hibrido.c -o mpi_02_hibrido.exe
 *  Ejecucion:    mpiexec -n 2 .\mpi_02_hibrido.exe
 *                mpiexec -n 4 .\mpi_02_hibrido.exe
 * ============================================================
 */

#include <mpi.h>
#include <omp.h>
#include <stdio.h>

int main(int argc, char** argv) {
    /* MPI_Init_thread en lugar de MPI_Init: necesario cuando se
       combina MPI con OpenMP. MPI_THREAD_FUNNELED indica que solo
       el hilo master de cada proceso (omp thread 0) hara llamadas
       MPI. Esto es suficiente para este ejercicio porque las
       llamadas MPI estan FUERA de las regiones paralelas. */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* TODO 1: Region paralela OpenMP con 4 hilos.
       Cada hilo se identifica imprimiendo el rank del proceso
       que lo contiene y su tid (ID de hilo). El orden no es
       deterministico ni entre hilos ni entre procesos. */
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();        /* ID del hilo (0..3) */
        int nthreads = omp_get_num_threads();  /* Total de hilos en la region */
        printf("  Proceso MPI %d │ Hilo OpenMP %d de %d\n",
               rank, tid, nthreads);
    }

    /* TODO 2: Solo el maestro imprime el total de unidades de
       computo activas. Lo hacemos FUERA de la region paralela
       para que solo se imprima una vez. */
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
