/*
 * ============================================================
 *  Ejercicio 3 - Suma hibrida de vector
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


    // TODO 1: Solo rank==0 crea y llena el arreglo completo 
    // arr[i] = i  para i en [0, N) 
    // Cada proceso recibe su porción del arreglo 

    if (rank == 0) {
        arr = (long long*) malloc(N * sizeof(long long));
        if (arr == NULL) {
            fprintf(stderr, "Error: malloc fallo en rank 0\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < N; i++) {
            arr[i] = i;
        }
    }

    long long *local = (long long*) malloc(chunk * sizeof(long long));
    if (local == NULL) {
        fprintf(stderr, "Error: malloc fallo en rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }


    // TODO 2: Usar MPI_Scatter para repartir 'arr' en 'local' 
    // Tipo: MPI_LONG_LONG 

    MPI_Scatter(arr,   chunk, MPI_LONG_LONG,
                local, chunk, MPI_LONG_LONG,
                0, MPI_COMM_WORLD);


    // TODO 3: Cada proceso suma su 'local' usando OpenMP 
    // Usar #pragma omp parallel for reduction(+:suma_local) 

    long long suma_local = 0;
    #pragma omp parallel for reduction(+:suma_local)
    for (int i = 0; i < chunk; i++) {
        suma_local += local[i];
    }


    // TODO 4: Usar MPI_Reduce para sumar todas las suma_local 
    // en suma_total (solo en rank 0), operación MPI_SUM 

    long long suma_total = 0;
    MPI_Reduce(&suma_local, &suma_total, 1,
               MPI_LONG_LONG, MPI_SUM,
               0, MPI_COMM_WORLD);

    /* Solo el maestro reporta el resultado y verifica. */
    if (rank == 0) {
        printf("Suma total = %lld\n", suma_total);
        printf("Esperado   = %lld\n", (long long)N * (N - 1) / 2);
        free(arr);
    }

    free(local);
    MPI_Finalize();
    return 0;
}
