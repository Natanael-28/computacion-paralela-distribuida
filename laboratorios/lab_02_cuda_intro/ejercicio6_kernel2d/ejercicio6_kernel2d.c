%%writefile ejercicio6_kernel2d.cu
#include <stdio.h>
#include <cuda_runtime.h>

#define FILAS 4
#define COLS 5

__global__ void inicializarMatrizLineal(int *d_mat, int filas, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    if (fila < filas && col < cols) {
        int idx = fila * cols + col;
        d_mat[idx] = idx;
    }
}

__global__ void inicializarMatrizSuma(int *d_mat, int filas, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    if (fila < filas && col < cols) {
        int idx = fila * cols + col;
        d_mat[idx] = fila + col;
    }
}

void imprimir(int *h_mat) {
    for (int i = 0; i < FILAS; i++) {
        for (int j = 0; j < COLS; j++)
            printf("%3d ", h_mat[i * COLS + j]);
        printf("\n");
    }
}

int main() {
    int h_mat[FILAS * COLS];
    int *d_mat;
    cudaMalloc((void**)&d_mat, FILAS * COLS * sizeof(int));

    dim3 hilosPorBloque(COLS, FILAS);
    dim3 numBloques(1, 1);

    inicializarMatrizLineal<<<numBloques, hilosPorBloque>>>(d_mat, FILAS, COLS);
    cudaDeviceSynchronize();
    cudaMemcpy(h_mat, d_mat, FILAS * COLS * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Matriz con indice lineal (i*COLS + j):\n");
    imprimir(h_mat);

    inicializarMatrizSuma<<<numBloques, hilosPorBloque>>>(d_mat, FILAS, COLS);
    cudaDeviceSynchronize();
    cudaMemcpy(h_mat, d_mat, FILAS * COLS * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nMatriz con suma de indices (i + j):\n");
    imprimir(h_mat);

    cudaFree(d_mat);
    return 0;
}