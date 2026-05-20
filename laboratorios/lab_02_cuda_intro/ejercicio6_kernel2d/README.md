# Ejercicio 6 — Kernel 2D: Inicialización de Matriz

**Categoría:** Kernels Básicos GPU — 2D

## Descripción

Este programa utiliza kernels con **organización bidimensional de hilos** para inicializar una matriz `4×5`. En lugar de usar una grilla y bloques 1D, se aprovecha el tipo `dim3` de CUDA, que permite organizar los hilos en estructuras 2D que reflejan la geometría natural del problema. Este patrón es la base de operaciones sobre imágenes y matrices en CUDA.

### Flujo del programa

1. Se reserva en GPU un arreglo de `FILAS × COLS = 20` enteros (la matriz se almacena como un arreglo 1D en row-major order).
2. Se configura el lanzamiento del kernel con `dim3`:
   - `dim3 hilosPorBloque(COLS, FILAS)` → un bloque de `5×4 = 20` hilos.
   - `dim3 numBloques(1, 1)` → un solo bloque.
   - Cada hilo del bloque corresponde a exactamente una celda de la matriz.
3. **Kernel 1 — Índice lineal**: cada hilo calcula su `(fila, col)`, deriva el índice lineal `idx = fila * cols + col` y escribe ese valor en `d_mat[idx]`. El resultado es la matriz con los enteros `0..19` ordenados en row-major.
4. **Kernel 2 — Suma de índices**: el mismo arreglo en GPU se reutiliza (in-place) y se sobreescribe con `d_mat[idx] = fila + col`. El resultado es una matriz donde cada celda contiene la suma de sus coordenadas.
5. Se transfieren los resultados de vuelta al host y se imprimen ambas matrices.

### Cálculo de índices 2D

Para una grilla y bloques 2D, cada hilo deriva su posición global así:

```c
int col  = blockIdx.x * blockDim.x + threadIdx.x;
int fila = blockIdx.y * blockDim.y + threadIdx.y;
```

Y para acceder a la celda `(fila, col)` en un arreglo lineal 1D que representa la matriz, se usa la fórmula de row-major:

```c
int idx = fila * COLS + col;
```

### Conceptos clave demostrados

- Uso de `dim3` para configurar grillas y bloques en 2D.
- Cálculo de índices 2D combinando `blockIdx.x/y`, `blockDim.x/y` y `threadIdx.x/y`.
- Convención row-major para representar matrices 2D como arreglos 1D en memoria.
- Reutilización del mismo arreglo de GPU entre dos lanzamientos de kernel consecutivos.
- Convención de mapeo: `x → columnas`, `y → filas` (es estándar en CUDA porque alinea hilos consecutivos sobre columnas consecutivas, lo que favorece accesos coalescidos a memoria).

### Configuración del lanzamiento

| Parámetro | Valor |
|-----------|-------|
| Dimensión del bloque | `(5, 4)` (5 columnas × 4 filas) |
| Número de bloques | `(1, 1)` |
| Hilos por bloque | 20 |
| Hilos totales | 20 (uno por celda) |

## Comandos

Ejecutado en Google Colab con GPU T4 (compute capability 7.5).

### Compilación
```bash
nvcc ejercicio6_kernel2d.cu -o ejercicio6_kernel2d -arch=sm_75
```

### Ejecución
```bash
./ejercicio6_kernel2d
```

## TAREA resuelta

> **Modifica el kernel para que `mat[i][j] = i + j` en vez del índice lineal.**

En lugar de simplemente reemplazar el kernel original, se implementaron **ambas versiones** como kernels separados (`inicializarMatrizLineal` e `inicializarMatrizSuma`) y se ejecutan secuencialmente sobre el mismo arreglo en GPU. Esto permite comparar visualmente las dos asignaciones en la misma corrida del programa: la primera matriz muestra el orden de memoria (índice lineal `i*COLS + j` que va de `0` a `19`), mientras que la segunda muestra la suma de coordenadas (la diagonal principal contiene `0`, la antidiagonal contiene valores crecientes hacia la esquina inferior derecha donde `i + j = 3 + 4 = 7`). El segundo kernel sobreescribe in-place los valores que dejó el primero, lo cual también demuestra que múltiples kernels pueden encadenarse sobre el mismo buffer en VRAM sin necesidad de liberarlo y volverlo a reservar.

## Evidencia

![Compilación y ejecución del Ejercicio 6](../img/ej6_evidencia.png)