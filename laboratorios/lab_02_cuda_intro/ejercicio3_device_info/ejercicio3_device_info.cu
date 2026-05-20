#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    printf("GPUs CUDA disponibles en este sistema: %d\n\n", numGPUs);

    for (int i = 0; i < numGPUs; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("=== GPU %d: %s ===\n", i, prop.name);
        printf("  Compute Capability     : %d.%d\n", prop.major, prop.minor);
        printf("  Memoria Global         : %.2f GB\n",
               (float)prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
        printf("  Memoria Compartida/Blq : %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Hilos maximos/Bloque   : %d\n", prop.maxThreadsPerBlock);
        printf("  Hilos maximos/SM       : %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Multiprocessors (SM)   : %d\n", prop.multiProcessorCount);
        printf("  Frecuencia del reloj   : %.2f GHz\n", prop.clockRate / 1e6f);
        printf("  Ancho de bus de memoria: %d bits\n", prop.memoryBusWidth);
        printf("  Dim. maxima de bloque  : (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Dim. maxima de grilla  : (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

        int totalHilos = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
        printf("  Total hilos residentes : %d\n", totalHilos);
        printf("\n");
    }
    return 0;
}
