# Entrenamiento de Redes Neuronales en GPU
* CUDA con PyTorch en Google Colab

---

| | |
|---|---|
| **Parcial** | Segundo Corte |
| **Materia** | Programación Paralela y Computación Distribuida |
| **Profesor** | Alejandro Jaimes|
| **Integrantes** |Josue Natanael — Marlon Arévalo |
| **Fecha** | |

---

## 0. Instrucciones Generales
* El taller se desarrolla en Google Colab usando una GPU gratuita de NVIDIA.
* Se trabaja en parejas; ambos integrantes deben entender cada parte.
* Se deben capturar pantallazos de cada salida importante indicada con [PANTALLAZO].
* Al finalizar, se descarga el notebook y se sube todo a un repositorio de GitHub.

### Preguntas
1. ¿Qué diferencia hay entre un notebook en la nube (Colab) y un entorno local como el del tutorial de instalación? ¿Cuál prefieren y por qué?



#### La principal diferencia radica en la **comodidad frente al control**:

* **Google Colab (Nube):** Es la opción rápida y sin fricciones. Te da acceso a una GPU gratis y no requiere instalar nada (*cero setup*). Sin embargo, tiene limitaciones: tus datos se suben a los servidores de Google, la sesión se desconecta por inactividad y dependes de su entorno preconfigurado.
* **Entorno Local:** Te da control total y privacidad absoluta (tus datos nunca salen de tu máquina), además de que puedes dejarlo corriendo el tiempo que quieras sin que se corte. El "precio" a pagar es que necesitas tener un buen hardware (una GPU de NVIDIA) y hacer toda la instalación técnica tú mismo (CUDA, PyTorch, etc.).

#### Nuestra preferencia y el porqué

Para este taller en particular, **preferimos usar Google Colab**.

**¿Por qué?** Porque nuestra computadora local tiene recursos muy limitados (solo 8 GB de RAM y una tarjeta gráfica integrada o iGPU). Intentar entrenar modelos usando solo el procesador (CPU) tardaría una eternidad; 


---

2. Antes de comenzar, hagan una predicción: ¿cuántas veces más rápida creen que será la GPU comparada con la CPU en el entrenamiento? Anoten su predicción aquí y compárenla al final con el resultado real.

>Predicción del grupo antes de correr el entrenamiento: la GPU será aprox. **9×** más rápida que la CPU.
>
>
---

## 1. Configurar el Entorno en Google Colab
* Activar la GPU desde el menú de Colab: Entorno de ejecución > Cambiar tipo de entorno de ejecución.
* Verificar que PyTorch reconoce la GPU y mostrar el nombre del dispositivo.
* Ejecutar `nvidia-smi` para ver el estado de la GPU, igual que en el tutorial de instalación.

### Preguntas
1. La salida de `nvidia-smi` muestra campos como *Driver Version*, *Memory Usage* y *GPU-Util*. ¿Qué indica cada uno?

- **Driver Version**: versión del driver de NVIDIA que comunica el sistema operativo con la GPU física. Define qué CUDA toolkit es compatible.
- **Memory Usage**: VRAM ocupada / VRAM total de la GPU. La T4 tiene ~15 GB. Al inicio está casi vacía; sube al mover el modelo y los batches.
- **GPU-Util**: porcentaje de uso del chip de cómputo (los núcleos CUDA). 0% en reposo, sube durante `forward`/`backward`.

2. Cuando activan el acelerador en Colab, ¿qué creen que ocurre físicamente? ¿La GPU está en su computador o en otro lugar? Propongan una analogía con algo de la vida cotidiana.

La GPU no está en nuestro computador. Google levanta una máquina virtual en uno de sus datacenters y nuestro notebook se conecta a ella vía un kernel de Jupyter remoto. Analogía: es como rentar un carro deportivo por hora — no es nuestro, solo lo usamos mientras dura la sesión.

3. `torch.cuda.is_available()` retorna `True` o `False`. ¿Qué condiciones deben cumplirse para que retorne `True`? Listen al menos tres requisitos.

El runtime de Colab tiene una GPU asignada (activada desde *Entorno de ejecución → Cambiar tipo de entorno*).

El driver NVIDIA está cargado en la máquina virtual y la GPU es visible al SO.

PyTorch fue compilado con soporte CUDA (las builds que Colab incluye por defecto sí lo traen).

---

## 2. Conceptos: CPU vs GPU en PyTorch
* Comparar las operaciones de CUDA en C con su equivalente en PyTorch.
* Entender cómo se mueven tensores entre CPU y GPU con `.to('cuda')`.
* Definir el dispositivo al inicio del proyecto para que el código funcione con o sin GPU.

### Preguntas
1. En el tutorial anterior usaron `cudaMemcpy` para mover datos entre CPU y GPU. En PyTorch eso se hace con `.to('cuda')`. ¿Qué ventaja le ven a la forma de PyTorch? ¿Qué se pierde al abstraerlo tanto?

La ventaja es la abstracción: el mismo código corre en CPU o GPU sin tocar nada — PyTorch despacha el kernel correcto según `tensor.device`. No escribimos kernels, no manejamos `gridDim` / `blockDim`, no hacemos `cudaFree`. Lo que se pierde es control fino: no podemos elegir el patrón de acceso a memoria compartida, ni el número exacto de hilos por bloque, ni hacer optimizaciones específicas del hardware. Para el 99% de los casos de deep learning eso no importa; para implementar un algoritmo no estándar (un simulador físico custom), CUDA C sigue siendo necesario.


2. Diagramen en Excalidraw el flujo de un tensor desde que se crea en CPU hasta que se opera en GPU y el resultado vuelve a CPU. Etiqueten cada flecha con la operación de PyTorch correspondiente.

![Flujo tensor](pantallazos/excalidraw_1.png)

3. ¿Por qué es una buena práctica usar la variable `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` en lugar de escribir `'cuda'` directamente en el código?



**Portabilidad**. Si el código corre en una máquina sin GPU (laptop, CI/CD, ambiente de pruebas), no falla — cae a CPU automáticamente. Centraliza la decisión en una sola línea: si mañana queremos cambiar a `cuda:1`, a `mps` (Mac M1/M2) o a múltiples GPUs, tocamos solo esa variable. Hardcodear `'cuda'` rompe el código en cualquier máquina sin GPU.

---

## 3. Preparar los Datos: Dataset MNIST

* Descargar el dataset MNIST: 60,000 imágenes de entrenamiento y 10,000 de prueba.
* Aplicar transformaciones para convertir las imágenes a tensores y normalizarlas.
* Visualizar una muestra del dataset para entender qué se va a clasificar.

### [PANTALLAZO 3] — Visualización del dataset
![Pantallazo 3](pantallazos/pantallazo3.png)

### Preguntas
1. El dataset se divide en 60,000 imágenes de entrenamiento y 10,000 de prueba. ¿Por qué no se entrena con todas las 70,000? Propongan una analogía con estudiar para un examen.

Porque necesitamos un conjunto de **prueba** que el modelo **jamás haya visto** para medir generalización vs memorización. Analogía con estudiar: si el profesor te entrega el examen final completo para practicar, sacar 5.0 en ese mismo examen no demuestra que aprendiste — solo que memorizaste. Las 10,000 imágenes de test son el "examen sorpresa" con problemas que nunca practicaste; la nota ahí sí mide aprendizaje real.

2. El `DataLoader` carga los datos en lotes (*batches*) de 64 imágenes. ¿Por qué no se pasan todas las imágenes de una sola vez a la GPU? Relacionen su respuesta con el concepto de memoria que vieron en `nvidia-smi`.

Memoria: aunque 60,000 imágenes (~190 MB) cabrían crudas en VRAM, las activaciones intermedias del forward (las salidas de cada capa, que se guardan para el backward) multiplican el consumo entre 10× y 100×. Hacer un solo batch gigantesco genera out-of-memory. En nvidia-smi vemos exactamente el límite de VRAM disponible.
Aprendizaje: con batch_size=64 la red ajusta sus pesos ~938 veces por época (60,000 / 64). Con un solo batch ajustaría una sola vez. Más actualizaciones → mejor convergencia y mejor generalización (el ruido del minibatch actúa como regularizador).

3. Cada imagen tiene forma `[1, 28, 28]`. Diagramen en Excalidraw qué representa cada dimensión y cómo luce ese tensor visualmente.

![Flujo tensor](pantallazos/excalidraw_2.png)

Forma [1, 28, 28]:

1: canal de color (escala de grises = 1 canal; RGB tendría 3).

28: alto de la imagen en píxeles.

28: ancho de la imagen en píxeles.

---

## 4. Construir la Red Neuronal
* Definir la arquitectura: capa de entrada (784), dos capas ocultas (256 y 128), capa de salida (10 dígitos).
* Mover el modelo a la GPU con `.to(device)`.
* Contar el total de parámetros entrenables de la red.

### [PANTALLAZO 4] 
![Pantallazo 4](pantallazos/pantallazo4.png)

### Preguntas
1. Diagramen en Excalidraw la arquitectura completa de la red: entrada → capa 1 → capa 2 → salida. Indiquen el número de neuronas en cada capa y qué función de activación se usa entre ellas.

### Diagrama de arquitectura (Excalidraw)
![Arquitectura red](pantallazos/excalidraw_3.png)


2. ¿Por qué la capa de entrada tiene exactamente 784 neuronas y la de salida exactamente 10? ¿Qué pasaría si pusieran 11 neuronas en la salida?

- **784 = 28 × 28**: cada píxel de la imagen aplanada es una neurona de entrada. Aplanamos la matriz 28×28 en un vector de 784 con `x.view(-1, 784)`.
- **10**: hay 10 clases posibles (los dígitos 0,1,...,9). Cada neurona de salida representa la *puntuación logit* para una clase. Si pusiéramos 11 neuronas, la onceava clase nunca recibiría gradiente (porque ninguna etiqueta apunta a ella en `CrossEntropyLoss`), sus pesos quedarían en sus valores aleatorios iniciales y sería ruido inútil que ralentiza el cómputo.

3. Cuando hacen `modelo.to(device)`, ¿qué creen que se está transfiriendo a la GPU? ¿Es solo el código, o algo más? Propongan una analogía con el tutorial de CUDA en C.

Se transfieren los **tensores de parámetros entrenables**: las matrices de pesos `W` y los vectores de bias `b` de cada capa. NO se transfiere código Python — eso siempre se interpreta en CPU. Cuando llamamos `modelo(x)`, Python en CPU orquesta las llamadas, pero cada operación matricial despacha un kernel CUDA que ejecuta los hilos en la GPU.

Analogía con CUDA C: equivale a hacer `cudaMalloc(&W_d, ...) + cudaMemcpy(W_d, W_h, ..., H2D)` para *cada* peso del modelo, todo en una sola llamada `.to(device)`.

**Parámetros totales:** 784·256 + 256 + 256·128 + 128 + 128·10 + 10 = **235,146 parámetros**.


---

## 5. Entrenar el Modelo: CPU vs GPU
* Entrenar el mismo modelo dos veces: primero en CPU, luego en GPU.
* Medir el tiempo de entrenamiento en cada dispositivo.
* Comparar los resultados y calcular cuántas veces más rápida fue la GPU.

### [PANTALLAZO 5] — Entrenamiento en CPU
![Pantallazo 4](pantallazos/pantallazo5.png)

### [PANTALLAZO 6] — Entrenamiento en GPU
![Pantallazo 5](pantallazos/pantallazo6.png)

### [PANTALLAZO 7] — Comparación final
![Pantallazo 6](pantallazos/pantallazo7.png)

### Preguntas
1. Registren aquí los tiempos obtenidos. ¿El resultado coincidió con la predicción que hicieron en la sección 0? ¿Qué los sorprendió?

En Colab con T4 el speedup típico para esta arquitectura (MLP pequeña, batch=64, 3 épocas) está entre **5× y 15×**. Si predijeron 9×, el resultado va a parecer modesto — pero es porque la red es chica; con CNNs grandes y batches de 256+ el speedup escala fácilmente a 50-100×. 

2. El entrenamiento repite el ciclo: *predicción → error → ajuste de pesos*. Propongan una analogía con algo cotidiano que siga el mismo ciclo de mejora por repetición.

Aprender a tirar al arco en básquetbol. Primer tiro (*predicción*): la pelota se queda corta. Sientes el error (*loss*): "me faltó fuerza". Ajustas el impulso para el próximo tiro (*backward + optimizer.step*). Repites mil veces y la mano sola sabe la fuerza exacta — los "pesos" musculares ya están entrenados. Otras analogías válidas: aprender a parquear un carro, afinar una receta, calibrar el volumen del despertador en una habitación nueva.


3. ¿Por qué creen que la GPU es más rápida en esta tarea? Relacionen su respuesta con el concepto de hilos y bloques que vieron en el tutorial de CUDA en C.

Porque cada paso del entrenamiento es **multiplicación masiva de matrices**. Calcular `imagenes @ W1` para un lote de 64 imágenes involucra 64 × 784 × 256 ≈ **12 millones de multiplicaciones independientes** entre sí. Una CPU las resuelve en secuencia con 8-16 cores. Una GPU tiene **miles de núcleos CUDA** que las hacen en paralelo, organizados exactamente en los **bloques** e **hilos** del tutorial de CUDA C: cada hilo computa un elemento del resultado, los hilos se agrupan en bloques (warps de 32 hilos), y el kernel `matmul` recorre la matriz entera en una sola pasada. La aritmética es la misma; lo que cambia es la cantidad de cores trabajando simultáneamente.

### Análisis de la Curva de Aprendizaje

Antes de responder, observen su gráfica generada y usen esta escala para interpretar el Loss:

| Loss final | Interpretación |
|---|---|
| 1.0 o más | La red no aprendió nada, está adivinando al azar |
| 0.3 - 0.5 | Aprendiendo, pero todavía comete muchos errores |
| 0.1 - 0.2 | Bien, la red entiende el problema |
| 0.07 o menos | Muy bien, la red generaliza correctamente |
| 0.01 o menos | Casi perfecto |

**Analogía:** el Training loss son los errores practicando con ejercicios del libro que ya conocen. El Test loss son los errores en el examen real, con preguntas que nunca vieron. Al inicio la red falla mucho con los ejercicios porque no sabe nada, pero como tampoco ha memorizado nada raro, falla de forma pareja en el examen. Conforme avanza, domina los ejercicios y eso se traduce en mejora en el examen real — ahí es donde las dos líneas convergen.

### Preguntas

1. Según la escala, ¿en qué rango quedó el Loss final de su modelo? ¿Lo consideran un buen resultado para 3 épocas? Justifiquen con base en la gráfica que generaron.

Según la tabla: con 3 épocas en GPU el Test loss típicamente queda entre 0.08 y 0.12, que cae en "Bien, la red entiende el problema" o cerca de "Muy bien". Para solo 3 épocas y una red pequeña es un buen resultado — confirma que MNIST es un problema fácil resoluble con MLPs.

2. Observen en qué época convergen las dos líneas. ¿Qué creen que pasaría si entrenaran 2 épocas más — el loss seguiría bajando indefinidamente o en algún punto se detendría? ¿Qué riesgo aparece si se entrena demasiado?

El Training loss seguiría bajando casi indefinidamente (la red puede memorizar las 60,000 imágenes). El Test loss baja, se estanca, y eventualmente sube otra vez — ese es el riesgo del overfitting (sobreajuste). La red comienza a memorizar idiosincrasias de los datos de entrenamiento que no generalizan. La práctica estándar es early stopping: detener el entrenamiento cuando el test loss deja de mejorar durante N épocas consecutivas

---

## 6. Evaluar y Visualizar Resultados
* Calcular la precisión del modelo sobre los datos de prueba que nunca vio durante el entrenamiento.
* Visualizar predicciones reales con indicadores de acierto (verde) y error (rojo).

### [PANTALLAZO 8] 
![Pantallazo 8](pantallazos/pantallazo8.png)


### [PANTALLAZO 9] 
![Pantallazo 9](pantallazos/pantallazo9.png)

### Preguntas
1. ¿Por qué la precisión se mide sobre datos que el modelo nunca vio durante el entrenamiento y no sobre los mismos datos con los que aprendió?

Porque queremos medir generalización, no memorización. Sobre los datos de entrenamiento la red puede llegar a 99-100% incluso si sobreajusta — esa nota no predice nada del comportamiento en el mundo real. El test set simula al usuario final subiendo dígitos que la red jamás vio. Si la precisión de test es mucho menor que la de train, hay sobreajuste; si son similares, el modelo generaliza bien.

2. Observen los dígitos que el modelo clasificó mal. ¿Tienen algo en común? ¿Por qué creen que la red se equivocó en esos casos específicos?

Casos típicos de fallo en MNIST:
- **4 vs 9**: un 4 con el bucle superior cerrado se parece a un 9.
- **3 vs 5 vs 8**: trazos con bucles laterales pueden confundirse.
- **7 vs 1**: 7 sin barra horizontal central queda como 1 con sombrero.
- **Trazos muy gruesos o muy delgados** respecto al promedio del dataset.

En general son dígitos **ambiguos donde incluso un humano dudaría**.


3. Si quisieran mejorar la precisión del modelo, ¿qué cambiarían de la arquitectura o del entrenamiento? Propongan al menos dos modificaciones y justifiquen cada una.


Dos modificaciones para mejorar la precisión:**

- **Usar una red convolucional (CNN) en lugar de MLP.** Las capas Conv2D explotan la estructura espacial 2D de la imagen (vecindad de píxeles, invarianza a traslación). Una CNN simple (Conv → MaxPool → Conv → MaxPool → FC) lleva MNIST de ~97% a ~99.5%. La MLP pierde información al aplanar la imagen en un vector — la relación espacial entre píxeles vecinos se "diluye" porque la red debe re-aprenderla con pesos sueltos.

- **Data augmentation + más épocas (5-10).** Generar variantes artificiales de cada imagen (rotaciones de ±10°, traslaciones pequeñas, escalado leve) usando `torchvision.transforms.RandomAffine`. Esto multiplica el dataset efectivo y obliga a la red a aprender características robustas en lugar de memorizar trazos específicos. Especialmente útil para que el modelo se comporte mejor con dígitos dibujados a mano fuera del dominio MNIST (como en la Sección 7).
---

## 7. Prueba tu Propio Dígito
* Dibujar un dígito del 0 al 9 en Paint (o cualquier editor), guardarlo como imagen.
* Subir la imagen a Colab y preprocesarla para que tenga el mismo formato que MNIST: escala de grises, fondo negro, trazo blanco, tamaño 28x28.
* Pasarla al modelo entrenado y ver qué predice.
* Visualizar la imagen tal como la ve la red antes de hacer la predicción.

### [PANTALLAZO 10] — Predicción del dígito propio
![Pantallazo 10](pantallazos/10_digito_propio.png)



### Preguntas
1. ¿El modelo acertó con tu dígito dibujado a mano? Si falló, ¿por qué creen que se equivocó? Comparen su imagen con las del dataset MNIST — ¿se ven similares o muy diferentes?

Si falló, las causas típicas son trazo muy delgado, dígito no centrado, fondo con ruido, o que el dígito escrito difiere mucho del estilo del dataset. Comparando con las muestras de la Sección 3 — los dígitos MNIST son **uniformemente gruesos, centrados, sobre fondo limpio**.

2. El preprocesamiento invierte los colores de la imagen (`ImageOps.invert`). ¿Por qué es necesario hacer eso antes de pasarla al modelo? ¿Qué pasaría si no se hiciera?

MNIST tiene **fondo negro (píxel ≈ 0) con trazo blanco (píxel ≈ 1)**. En Paint dibujamos al revés: fondo blanco con trazo negro. Si no invirtiéramos, la red vería casi todo el lienzo "activo" (porque el fondo blanco normalizado tiene valor positivo alto) y un trazo "negativo" en el centro. La red aprendió a reconocer trazos blancos sobre fondo negro — pasarle la imagen sin invertir equivale a pedirle leer un texto en negativo fotográfico.


3. Prueben con un dígito que crean que va a fallar — por ejemplo un 4 o un 9 escritos de forma poco convencional. ¿Falló? ¿Qué dice eso sobre las limitaciones del modelo entrenado solo con MNIST?

un **4 con bucle superior cerrado** se confunde con **9**, o que un **1 con base larga** se confunda con **2** o **7**. Esto evidencia que MNIST es un dataset **limpio y estilizado**: dígitos centrados, trazo grueso uniforme, fondo nítido. En el mundo real (fotos de cheques, formularios escaneados con ruido, dígitos en perspectiva), un modelo entrenado solo en MNIST no alcanza — se necesitan datasets más diversos como SVHN, EMNIST, o data augmentation agresiva.

4. Tomar captura, de almenos una predicción que se haya hecho correctamente.



---

### Bonus: ¿Qué tan seguro está el modelo?

Hasta ahora sabemos *qué* predice el modelo, pero no *qué tan seguro* está de su respuesta. Un modelo puede predecir "7" con un 95% de confianza o con un 40% — y eso hace toda la diferencia.

Ejecuten la siguiente celda para ver la distribución de probabilidades sobre los 10 dígitos para ambos modelos. Si el modelo está seguro, un dígito tendrá un porcentaje muy alto y los demás estarán cerca de 0. Si está dudando, verán los porcentajes distribuidos entre varios dígitos.

```python
# Ver qué tan seguro está cada modelo
import torch.nn.functional as F

with torch.no_grad():
    # GPU
    tensor_gpu = transform(imagen).unsqueeze(0).to('cuda')
    salida_gpu = modelo_gpu(tensor_gpu)
    prob_gpu = F.softmax(salida_gpu, dim=1)[0]
    
    # CPU
    tensor_cpu = transform(imagen).unsqueeze(0).to('cpu')
    salida_cpu = modelo_cpu(tensor_cpu)
    prob_cpu = F.softmax(salida_cpu, dim=1)[0]

print("Probabilidades GPU:")
for i, p in enumerate(prob_gpu):
    print(f"  {i}: {p.item()*100:.1f}%")

print("\nProbabilidades CPU:")
for i, p in enumerate(prob_cpu):
    print(f"  {i}: {p.item()*100:.1f}%")
```

**Observen y respondan:**
1. ¿Cuál dígito tiene la probabilidad más alta en cada modelo? ¿Coincide con la predicción?

Con todos los que probamos el que tiene la probabilidad mas alta fue el 3

2. ¿El modelo está seguro o dudando? ¿Cómo lo saben mirando los porcentajes?

Cuando hay porcetajes muy bajos

3. Si el porcentaje más alto es menor al 50%, ¿confiarían en esa predicción? ¿Por qué?

No, por que sigue adivinando y no es un resultado seguro
---

El bloque de código lo reemplazas con la función completa que ya tenemos. ¿Lo agregamos también al markdown del taller?


## 8. Preguntas de Reflexión y Entregables
* Responder 4 preguntas que conectan lo aprendido en PyTorch con el tutorial de CUDA en C.
* Subir a GitHub el notebook descargado y un reporte en Markdown con pantallazos y respuestas.

### Preguntas
1. Ahora que completaron todo el taller, ¿en qué se parece PyTorch a programar en CUDA directamente y en qué se diferencia? ¿Cuándo usarían uno y cuándo el otro?

**Similitudes:**
- Ambos operan sobre la misma GPU física y eventualmente despachan kernels CUDA. PyTorch no inventa nada nuevo a nivel de hardware; usa cuBLAS, cuDNN y kernels CUDA precompilados.
- Ambos requieren mover datos explícitamente entre CPU y GPU (en CUDA C con `cudaMemcpy`, en PyTorch con `.to(device)`). El concepto de memoria separada host/device no desaparece, solo cambia de sintaxis.
- Ambos paralelizan en hilos y bloques; PyTorch simplemente ya tiene la configuración óptima para operaciones tensoriales comunes.

**Diferencias:**

| Aspecto | CUDA C | PyTorch |
|---|---|---|
| Abstracción | Hilos, bloques, memoria compartida | Tensores, módulos, autograd |
| Líneas de código para una matmul | ~100 (kernel + setup + memcpy) | 1 línea: `a @ b` |
| Curva de aprendizaje | Empinada | Suave |
| Control sobre el hardware | Total | Limitado a kernels predefinidos |
| Autograd | Manual (calcular derivadas a mano) | Automático |
| Casos de uso | HPC custom, simuladores físicos, embebidos (Jetson) | Deep learning, prototipado, investigación |

**¿Cuándo usar cuál?**
- **PyTorch** cuando el problema se expresa con operaciones tensoriales estándar (cualquier modelo de deep learning moderno). Es el 99% de los casos.
- **CUDA C** cuando: (a) hay un kernel custom no expresable en PyTorch (un *N-body simulator*, un solver de ecuaciones diferenciales específico), (b) trabajamos en embebidos con restricciones de tamaño, (c) queremos arrancarle el último 5-10% de rendimiento a un kernel crítico — en ese caso PyTorch incluso permite incrustar extensiones CUDA C++ vía `torch.utils.cpp_extension`.


2. Diagramen en Excalidraw el flujo completo del taller: desde la activación de la GPU hasta la predicción final. Úsenlo como resumen visual de todo lo que hicieron.

### [PANTALLAZO 11]
![Pantallazo 11](pantallazos/excalidraw_4.png)


3. Si tuvieran que explicarle este taller a alguien que nunca ha programado, ¿cómo describirían en una sola analogía lo que hace una red neuronal entrenándose en una GPU?

> Una red neuronal entrenándose en una GPU es como **una orquesta de mil músicos aprendiendo una canción nueva**. La GPU es la orquesta entera: miles de músicos (núcleos CUDA) tocando a la vez, cada uno ejecutando una parte minúscula de la partitura (una multiplicación). Una CPU sería un solista virtuoso: muy hábil, pero uno solo. Para canciones cortas el solista basta; para una sinfonía completa con sesenta mil compases (las imágenes de MNIST), necesitas la orquesta. El "entrenamiento" es el ensayo: la primera vez suenan terrible, el director (la función de pérdida) les indica los errores compás por compás, cada músico ajusta su técnica (sus pesos), y tras varios ensayos completos (épocas) tocan la canción casi perfecta — pueden incluso interpretar canciones nuevas que nunca ensayaron porque internalizaron el estilo.