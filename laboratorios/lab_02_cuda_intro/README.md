# Lab 02 — Introducción a CUDA

**Materia:** Programación Paralela y Computación Distribuida
**Semestre:** 2026-I
**Docente:** Prf. Juan Alejandro Carrillo Jaimes
**Universidad de Pamplona** — Facultad de Ingenierías y Arquitectura

## Descripción

Este laboratorio introduce la programación paralela sobre GPUs usando **CUDA** (Compute Unified Device Architecture) de NVIDIA. A través de 9 ejercicios de dificultad creciente se exploran los conceptos fundamentales del modelo de programación CUDA: la jerarquía de hilos, bloques y grillas, la gestión de memoria entre host y device, el lanzamiento de kernels, la sincronización, el uso de shared memory y la medición de rendimiento con CUDA Events.

## Entorno de ejecución

Como el equipo de desarrollo no cuenta con una GPU NVIDIA local, **todos los ejercicios se ejecutaron en Google Colab** con el siguiente entorno:

| Componente | Detalle |
|------------|---------|
| Plataforma | Google Colab (notebook gratuito) |
| GPU | NVIDIA Tesla T4 |
| Compute Capability | 7.5 (arquitectura Turing) |
| Compilador | `nvcc` (NVIDIA CUDA Compiler) |
| Flag de arquitectura | `-arch=sm_75` |

## Estructura del repositorio
lab_02_cuda_intro/
├── README.md                        ← este archivo
├── reporte/
│   └── taller_cuda.md               ← reporte completo con evidencias
├── img/                             ← pantallazos de todos los ejercicios
├── ejercicio1_hola_gpu/
│   ├── ejercicio1_hola_gpu.cu
│   └── README.md
├── ejercicio2_matriz/
│   ├── ejercicio2_matriz.cu
│   └── README.md
├── ejercicio3_device_info/
│   ├── ejercicio3_device_info.cu
│   └── README.md
├── ejercicio4_suma_vectores/
│   ├── ejercicio4_suma_vectores.cu
│   └── README.md
├── ejercicio5_cuadrado/
│   ├── ejercicio5_cuadrado.cu
│   └── README.md
├── ejercicio6_kernel2d/
│   ├── ejercicio6_kernel2d.cu
│   └── README.md
├── ejercicio7_reduccion/
│   ├── ejercicio7_reduccion.cu
│   └── README.md
├── ejercicio8_tiempo/
│   ├── ejercicio8_tiempo.cu
│   └── README.md
└── ejercicio9_producto_punto/
├── ejercicio9_producto_punto.cu
└── README.md
## Ejercicios

### Categoría 1 — Transferencia de Datos CPU ↔ GPU

| # | Ejercicio | Concepto principal |
|---|-----------|---------------------|
| 1 | [Hola GPU](ejercicio1_hola_gpu/) | Ciclo CPU → GPU → CPU con `cudaMemcpy` |
| 2 | [Matriz 2D](ejercicio2_matriz/) | Transferencia de matrices 2D como arreglos 1D |
| 3 | [Device Info](ejercicio3_device_info/) | Consulta de propiedades de la GPU |

### Categoría 2 — Kernels Básicos en GPU

| # | Ejercicio | Concepto principal |
|---|-----------|---------------------|
| 4 | [Suma de Vectores](ejercicio4_suma_vectores/) | Primer kernel real, paralelismo masivo (1M de hilos) |
| 5 | [Cuadrado in-place](ejercicio5_cuadrado/) | Operación in-place sobre arreglo en GPU |
| 6 | [Kernel 2D](ejercicio6_kernel2d/) | Configuración con `dim3`, indexación 2D |

### Categoría 3 — Nivel Intermedio

| # | Ejercicio | Concepto principal |
|---|-----------|---------------------|
| 7 | [Reducción](ejercicio7_reduccion/) | Shared memory + `__syncthreads()` + reducción en árbol |
| 8 | [Medición de Tiempo](ejercicio8_tiempo/) | CUDA Events, bandwidth, speedup GPU vs CPU |
| 9 | [Producto Punto](ejercicio9_producto_punto/) | Fusión de multiplicación + reducción en un solo kernel |

## Cómo reproducir los ejercicios

1. Abrir [Google Colab](https://colab.research.google.com) y crear un notebook nuevo.
2. Activar GPU: `Entorno de ejecución` → `Cambiar tipo de entorno de ejecución` → **GPU T4**.
3. Verificar disponibilidad de CUDA: `!nvcc --version` y `!nvidia-smi`.
4. Por cada ejercicio, ejecutar tres celdas:
```python
   %%writefile ejercicioN.cu
   <pegar contenido del archivo .cu>
```
```python
   !nvcc ejercicioN.cu -o ejercicioN -arch=sm_75
```
```python
   !./ejercicioN
```

## Reporte completo

El reporte consolidado con descripción, comandos, evidencias e interpretación de cada ejercicio está en [`reporte/taller_cuda.md`](reporte/taller_cuda.md).

