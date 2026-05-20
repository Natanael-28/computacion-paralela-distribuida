%%writefile ejercicio8_tiempo.cu
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 10000000
#define THREADS 256

__global__ void escalarMult(float *d_vec, float escalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_vec[idx] *= escalar;
}

void escalarMultCPU(float *vec, float escalar, int n) {
    for (int i = 0; i < n; i++) vec[i] *= escalar;
}

int main() {
    float escalar = 2.5f;
    size_t bytes = N * sizeof(float);

    float *h_vec = (float*)malloc(bytes);
    float *h_vec_cpu = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { h_vec[i] = 1.0f; h_vec_cpu[i] = 1.0f; }

    float *d_vec;
    cudaMalloc((void**)&d_vec, bytes);
    cudaMemcpy(d_vec, h_vec, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t inicio, fin;
    cudaEventCreate(&inicio);
    cudaEventCreate(&fin);

    int numBloques = (N + THREADS - 1) / THREADS;

    cudaEventRecord(inicio);
    escalarMult<<<numBloques, THREADS>>>(d_vec, escalar, N);
    cudaEventRecord(fin);
    cudaEventSynchronize(fin);

    float ms_gpu = 0;
    cudaEventElapsedTime(&ms_gpu, inicio, fin);
    printf("Tiempo GPU: %.4f ms\n", ms_gpu);

    float gb = (2.0f * bytes) / (1024.0f * 1024.0f * 1024.0f);
    float bandwidth = gb / (ms_gpu / 1000.0f);
    printf("Bandwidth efectivo: %.2f GB/s\n", bandwidth);

    clock_t t_ini = clock();
    escalarMultCPU(h_vec_cpu, escalar, N);
    clock_t t_fin = clock();
    float ms_cpu = 1000.0f * (float)(t_fin - t_ini) / CLOCKS_PER_SEC;
    printf("Tiempo CPU: %.4f ms\n", ms_cpu);
    printf("Speedup GPU vs CPU: %.2fx\n", ms_cpu / ms_gpu);

    cudaMemcpy(h_vec, d_vec, sizeof(float), cudaMemcpyDeviceToHost);
    printf("h_vec[0] = %.1f (esperado %.1f)\n", h_vec[0], escalar);

    cudaEventDestroy(inicio); cudaEventDestroy(fin);
    cudaFree(d_vec); free(h_vec); free(h_vec_cpu);
    return 0;
}