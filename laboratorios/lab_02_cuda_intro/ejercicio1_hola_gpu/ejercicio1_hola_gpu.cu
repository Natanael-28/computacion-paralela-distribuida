#include <stdio.h> 
#include <cuda_runtime.h> 

#define N 10 

  
int main() { 

    // 1. Crear arreglo en CPU (host) 

    int h_datos[N]; 
    int h_resultado[N]; 
    for (int i = 0; i < N; i++) h_datos[i] = i * 3; // Llenar con múltiplos de 3 

    // 2. Declarar puntero para GPU y reservar memoria 

    int *d_datos; 
    cudaMalloc((void**)&d_datos, N * sizeof(int)); 

    // 3. Copiar datos CPU → GPU 

    cudaMemcpy(d_datos, h_datos, N * sizeof(int), cudaMemcpyHostToDevice); 
    printf("Datos copiados a GPU correctamente.\n"); 

    // 4. Copiar datos GPU → CPU (de vuelta) 

    cudaMemcpy(h_resultado, d_datos, N * sizeof(int), cudaMemcpyDeviceToHost); 

    // 5. Verificar que los datos son correctos 

    printf("Verificacion de datos:\n"); 
    int ok = 1; 
    for (int i = 0; i < N; i++) { 
        printf("  h_datos[%d]=%d, h_resultado[%d]=%d", i, h_datos[i], i, h_resultado[i]); 
        if (h_datos[i] != h_resultado[i]) { printf(" *** ERROR ***"); ok = 0; } 
        printf("\n"); 
    } 

    printf(ok ? "\n[OK] Transferencia exitosa!\n" : "\n[FALLO] Hay errores.\n"); 

    // 6. Liberar memoria GPU 

    cudaFree(d_datos); 

    return 0; 

} 

  

// Compilar con: nvcc ejercicio1_hola_gpu.cu -o ejercicio1 

// Ejecutar con: .\ejercicio1.exe  (Windows) o  ./ejercicio1  (Linux) 

 
