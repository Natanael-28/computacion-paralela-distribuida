/*
 * ============================================================
 *  Ejercicio 1 - Hola Mundo MPI
 *  Autores: Josue Natanael Reyes & Marlon Arevalo
 * ============================================================
 *
  */
 
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    /* Inicializa el entorno MPI. Debe ser la primera llamada MPI. */
    MPI_Init(&argc, &argv);

    int rank, size;
    /* rank = ID unico de este proceso (0, 1, 2, ...).
       size = numero total de procesos lanzados con -n. */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* TODO 1: Cada proceso se identifica.
       El orden de aparicion en pantalla NO es deterministico. */
    printf("Proceso %d de %d: ¡Hola desde MPI!\n", rank, size);

    /* TODO 2: Solo el maestro (rank 0) imprime el mensaje final.
       Esto es un patron muy comun: dejar al rank 0 hacer la
       salida "global" mientras el resto solo trabaja. */
    if (rank == 0) {
        printf("[MAESTRO] Todos los %d procesos han saludado.\n", size);
    }

    /* Cierra el entorno MPI. Debe ser la ultima llamada MPI. */
    MPI_Finalize();
    return 0;
}
