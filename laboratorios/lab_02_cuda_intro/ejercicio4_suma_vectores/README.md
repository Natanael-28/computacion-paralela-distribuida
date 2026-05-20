# Ejercicio 4 — Suma de Vectores Paralela

Categoría: Kernels Básicos GPU

## Descripción

Este programa implementa la suma elemento a elemento de dos vectores de un millón de floats usando un kernel CUDA. Cada hilo de la GPU se encarga de sumar exactamente un par de elementos, lo que permite que el millón de operaciones se ejecute de forma masivamente paralela en lugar de secuencialmente como lo haría una CPU. Es el ejemplo canónico para entender el patrón completo de CUDA: reservar memoria, transferir datos, lanzar kernel, recuperar resultados y liberar memoria.

### Flujo del programa

1. Se reservan en CPU tres arreglos h_A, h_B y h_C de N = 1,000,000 floats con malloc.
2. h_A se llena con 1.0f y h_B con 2.0f, de modo que el resultado esperado en cada posición sea 3.0f.
3. Se reservan en GPU los tres arreglos correspondientes (`d_A`, d_B, d_C`) con `cudaMalloc.
4. Se copian h_A y h_B a la GPU con cudaMemcpyHostToDevice.
5. Se calcula la configuración de lanzamiento:
   numBloques = (N + THREADS_POR_BLOQUE - 1) / THREADS_POR_BLOQUE
   Esta fórmula es el patrón estándar de CUDA para garantizar que siempre haya suficientes hilos para cubrir todos los elementos, incluso cuando N no es divisible exactamente por el tamaño de bloque.
6. Se lanza el kernel sumaVectores<<<numBloques, THREADS_POR_BLOQUE>>>(d_A, d_B, d_C, N).
7. cudaDeviceSynchronize() bloquea la CPU hasta que la GPU termine, garantizando que los resultados estén listos antes de copiarlos.
8. Se copia d_C de regreso a h_C con cudaMemcpyDeviceToHost.
9. Se verifica el primer y último elemento, y se cuenta el número total de errores recorriendo todo el vector resultado.
10. Se libera tanto la memoria de GPU (`cudaFree`) como la de CPU (`free`).

### El kernel en detalle

__global__ void sumaVectores(float *d_A, float *d_B, float *d_C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_C[idx] = d_A[idx] + d_B[idx];
    }
}

- __global__ indica que la función se ejecuta en GPU pero se llama desde CPU.
- int idx = blockIdx.x * blockDim.x + threadIdx.x calcula el índice global único de cada hilo dentro de la grilla. Es la fórmula fundamental de CUDA en 1D.
- if (idx < n) es el guard: descarta los hilos sobrantes que la fórmula de bloques genera cuando N no es múltiplo exacto del tamaño de bloque. Sin este guard, los hilos extra leerían y escribirían fuera del arreglo y causarían comportamiento indefinido.
- Cada hilo realiza una sola operación: leer dos posiciones de memoria global y escribir una. El paralelismo proviene de que esto sucede simultáneamente para todos los hilos activos.

### Conceptos clave demostrados

- Declaración y lanzamiento de un kernel con la sintaxis <<<numBloques, hilosPorBloque>>>.
- Cálculo del índice global de un hilo en 1D.
- Patrón de "guard" para proteger accesos fuera de rango.
- Sincronización explícita CPU-GPU con cudaDeviceSynchronize().
- Ciclo completo de gestión de memoria: cudaMalloc → cudaMemcpy (H2D) → kernel → cudaMemcpy (D2H) → cudaFree.

## Comandos

Ejecutado en Google Colab con GPU Tesla T4 (compute capability 7.5).

### Compilación
nvcc ejercicio4_suma_vectores.cu -o ejercicio4

### Ejecución
./ejercicio4

## Salida obtenida

Lanzando 3907 bloques x 256 hilos = 1000192 hilos totales
h_C[0] = 3.0 (esperado: 3.0)
h_C[N-1] = 3.0 (esperado: 3.0)
Errores totales: 0

[OK] Suma de vectores completada.

Se observa que para procesar N = 1,000,000 elementos con bloques de 256 hilos se requieren (1,000,000 + 255) / 256 = 3907 bloques, lo que totaliza 3907 × 256 = 1,000,192 hilos lanzados. Los 192 hilos sobrantes son descartados por el if (idx < n) y no escriben en memoria.

## Evidencia

!Compilación y ejecución del Ejercicio 4
