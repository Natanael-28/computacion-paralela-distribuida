#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1000000
#define THREADS_POR_BLOQUE 256

__global__ void sumaVectores(float *d_A, float *d_B, float *d_C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_C[idx] = d_A[idx] + d_B[idx];
    }
}

int main() {
    size_t bytes = N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int numBloques = (N + THREADS_POR_BLOQUE - 1) / THREADS_POR_BLOQUE;
    printf("Lanzando %d bloques x %d hilos = %d hilos totales\n",
           numBloques, THREADS_POR_BLOQUE, numBloques * THREADS_POR_BLOQUE);
    sumaVectores<<<numBloques, THREADS_POR_BLOQUE>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    printf("h_C[0] = %.1f (esperado: 3.0)\n", h_C[0]);
    printf("h_C[N-1] = %.1f (esperado: 3.0)\n", h_C[N-1]);

    int errores = 0;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 3.0f) errores++;
    }
    printf("Errores totales: %d\n", errores);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    printf("\n[OK] Suma de vectores completada.\n");
    return 0;
}
