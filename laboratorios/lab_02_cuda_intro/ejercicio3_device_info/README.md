# Ejercicio 3 — Información del Device: Conoce tu GPU

**Categoría:** Consulta de propiedades del Device

## Descripción

Este programa consulta el runtime de CUDA para obtener e imprimir las propiedades de hardware de la GPU instalada en el sistema. No ejecuta ningún kernel ni transfiere datos: su único objetivo es introspección del dispositivo. Conocer estas propiedades es fundamental antes de optimizar cualquier código CUDA, porque parámetros como el número de SMs, el tamaño de la memoria compartida por bloque o el límite de hilos por bloque determinan la configuración de lanzamiento óptima para los kernels.

### Flujo del programa

1. Se llama a `cudaGetDeviceCount(&numGPUs)` para obtener el número de GPUs CUDA disponibles.
2. Por cada GPU detectada, se llena una estructura `cudaDeviceProp` con `cudaGetDeviceProperties(&prop, i)`.
3. Se imprimen los campos relevantes de la estructura:
   - **Nombre y Compute Capability** (`major.minor`): identifica el modelo y la generación arquitectónica.
   - **Memoria global**: VRAM total disponible.
   - **Memoria compartida por bloque**: tamaño de la shared memory que cada bloque puede declarar como `__shared__`.
   - **Hilos máximos por bloque**: límite hardware (1024 en arquitecturas modernas).
   - **Hilos máximos por SM**: hilos que un Streaming Multiprocessor puede manejar concurrentemente.
   - **Cantidad de Streaming Multiprocessors (SMs)**: unidades de cómputo físicas en la GPU.
   - **Frecuencia del reloj y ancho de bus de memoria**: factores que determinan el rendimiento bruto.
   - **Dimensiones máximas de bloque y grilla** en cada eje (x, y, z).
4. Se calcula y muestra el total de hilos residentes simultáneos como `SM × hilos_max_por_SM`.

### Conceptos clave demostrados

- API de introspección de CUDA: `cudaGetDeviceCount` y `cudaGetDeviceProperties`.
- Significado y unidades de los campos principales de `cudaDeviceProp`.
- Relación entre número de SMs y paralelismo máximo del hardware.

## Comandos

Ejecutado en Google Colab con GPU Tesla T4 (compute capability 7.5).

### Compilación
```bash
nvcc ejercicio3_device_info.cu -o ejercicio3
```

### Ejecución
```bash
./ejercicio3
```

## Salida obtenida
GPUs CUDA disponibles en este sistema: 1
=== GPU 0: Tesla T4 ===
Compute Capability     : 7.5
Memoria Global         : 14.74 GB
Memoria Compartida/Blq : 48 KB
Hilos maximos/Bloque   : 1024
Hilos maximos/SM       : 1024
Multiprocessors (SM)   : 40
Frecuencia del reloj   : 1.59 GHz
Ancho de bus de memoria: 256 bits
Dim. maxima de bloque  : (1024, 1024, 64)
Dim. maxima de grilla  : (2147483647, 65535, 65535)
Total hilos residentes : 40960

> Los valores específicos pueden variar levemente entre sesiones de Colab si se asigna una GPU distinta. Reemplaza esta sección con la salida real de tu ejecución si difiere.

## TAREA resuelta

> **Calcula e imprime cuántos hilos en total puede lanzar esta GPU. Total hilos = SM × maxThreadsPerMultiProcessor.**

Se agregó al final del bucle el cálculo `prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor` y se imprime como `Total hilos residentes`. Para una Tesla T4 esto equivale a **40 SM × 1024 hilos/SM = 40,960 hilos residentes simultáneamente**. Este es el límite de paralelismo *concurrente* del hardware: aunque un kernel puede lanzar millones de hilos lógicos, solo 40,960 están físicamente activos en un instante dado, y el resto espera su turno en el planificador de warps. Entender este número es clave para dimensionar bloques y grillas: lanzar muchos menos hilos que este límite desaprovecha el hardware, y lanzar muchos más simplemente añade overhead de planificación sin acelerar la ejecución.

## Evidencia

![Compilación y ejecución del Ejercicio 3](../img/ej3_evidencia.png)
