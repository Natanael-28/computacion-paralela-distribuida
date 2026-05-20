/*
 * ============================================================
 *  Ejercicio 3 - Suma hibrida de vector
 *  Lab 01 - Programacion Hibrida MPI + OpenMP
 *  Autores: Josue Natanael Reyes & Marlon Arevalo
 * ============================================================
 *
 *  Flujo:
 *    1. rank 0 inicializa arr[N] con arr[i] = i.
 *    2. MPI_Scatter reparte 'arr' en bloques de chunk = N/size
 *       elementos hacia el buffer 'local' de cada proceso.
 *    3. Cada proceso suma su 'local' en paralelo con OpenMP
 *       usando reduction(+:suma_local).
 *    4. MPI_Reduce combina las sumas parciales en suma_total
 *       (solo el rank 0 obtiene el resultado).
 *
 *  Verificacion: la suma 0+1+2+...+(N-1) = N*(N-1)/2
 *                Para N=1,000,000 -> 499,999,500,000
 *
 *  Compilacion:  mpicc -fopenmp mpi_03_suma_hibrida.c -o mpi_03.exe
 *  Ejecucion:    mpiexec -n 4 .\mpi_03.exe
 *
 *  IMPORTANTE: N debe ser divisible por el numero de procesos.
 *              Con N=1,000,000 funciona para -n 1, 2, 4, 5, 8, 10...
 *              Con -n 3 daria resultados erroneos.
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

    int chunk = N / size;       /* Elementos por proceso */
    long long *arr = NULL;      /* Solo rank 0 lo asignara */

    /* TODO 1: rank 0 crea y llena el arreglo completo.
       Los demas procesos NO necesitan tener arr completo,
       solo recibiran su porcion via MPI_Scatter. */
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

    /* Cada proceso reserva memoria para su porcion local. */
    long long *local = (long long*) malloc(chunk * sizeof(long long));
    if (local == NULL) {
        fprintf(stderr, "Error: malloc fallo en rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* TODO 2: MPI_Scatter divide 'arr' en 'size' bloques de
       'chunk' elementos y envia uno a cada proceso.
       - Buffer de envio (arr): solo importa en el root (rank 0).
       - Buffer de recepcion (local): cada proceso recibe sus
         'chunk' elementos aqui.
       - root = 0: el rank que tiene el arreglo completo. */
    MPI_Scatter(arr,   chunk, MPI_LONG_LONG,
                local, chunk, MPI_LONG_LONG,
                0, MPI_COMM_WORLD);

    /* TODO 3: Cada proceso suma su trozo local en paralelo con
       OpenMP. La clausula reduction(+:suma_local) crea una copia
       privada de suma_local por hilo, las suma de manera segura
       al final, y nos evita condiciones de carrera. */
    long long suma_local = 0;
    #pragma omp parallel for reduction(+:suma_local)
    for (int i = 0; i < chunk; i++) {
        suma_local += local[i];
    }

    /* TODO 4: MPI_Reduce combina las suma_local de TODOS los
       procesos aplicando MPI_SUM, y deja el resultado solo
       en suma_total del rank 0. */
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
