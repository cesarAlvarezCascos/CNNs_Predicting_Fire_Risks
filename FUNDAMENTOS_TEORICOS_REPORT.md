# Fundamentos teóricos para el report

Ideas, métodos y decisiones de diseño con base fundamentada, para citarlos en el report final.
Formato: idea breve + referencia. Ampliar al redactar.

\---

## 1\. Datos y preprocesado

* **Normalización con media/desviación de ImageNet**: centra cada canal (entrenamiento más estable) y es obligatoria para el fine-tuning (estadísticas con las que se preentrenó el modelo).
* **Aumentos que no cambian la clase**: en imágenes satelitales no hay orientación canónica → volteos y giros de 90° son "gratis" (sin pérdida de información). Cambios de color suaves (gamma, brillo, contraste, saturación) para no alterar señales de la vegetación.
* **El orden de las transformaciones importa** (demostrado en el notebook con la misma semilla y distinto orden de las mismas transformaciones).
* **RandomCrop en train / CenterCrop en val-test**: aleatoriedad solo en entrenamiento; evaluación determinista.
* **Augmentation en cada batch y cada época**: `RandomCrop(224)` en CPU (recorte distinto de la imagen 320×320 al leerla) + `GPUAugment` en GPU (volteos, giros 90° y color aleatorios por batch) → en cada época la red ve versiones distintas de cada imagen, aunque el train set sea el mismo.
  * Con semilla fija la secuencia aleatoria se repite entre ejecuciones de la misma configuración (las primeras épocas de la Fase 3 ven los mismos aumentos que en su fase): intencionado, por reproducibilidad.
* **Eficiencia**: imágenes precargadas en RAM (uint8) + aumentos por lotes en GPU → se eliminó el cuello de botella de la CPU (\~50 s/época antes).

## 2\. Arquitectura (CustomNet)

* **Bloques tipo VGG con convoluciones 3×3 apiladas**: dos 3×3 ven lo mismo que una 5×5 con menos parámetros y más no linealidad (Simonyan \& Zisserman, 2014).
* **BatchNorm tras cada convolución**: estabiliza y acelera el entrenamiento (Ioffe \& Szegedy, 2015).
* **Inicialización Kaiming (He)**: diseñada para ReLU, mantiene la varianza de las activaciones entre capas (He et al., 2015).
* **Global Average Pooling vs. Flatten**: con Flatten, la primera capa FC concentraba \~99 % de los parámetros (50 176 entradas) → sobreajuste y entrenamiento inestable. GAP reduce drásticamente parámetros y actúa como regularizador (Lin et al., 2013, *Network in Network*).

  * Evidencia propia: en la Fase 1 (máq. 0) las 6 configuraciones con Flatten quedaron por debajo de todas las de GAP.
* **Dropout** antes de la capa de salida como regularización (Srivastava et al., 2014).
* **Logits + CrossEntropyLoss**: la red no aplica softmax; la pérdida lo integra (numéricamente estable). Softmax solo para generar probabilidades de test.

## 3\. Entrenamiento

* **AdamW**: weight decay desacoplado del gradiente, regularización más correcta que la L2 de Adam (Loshchilov \& Hutter, 2019).
* **LR con warmup + decaimiento coseno**: warmup evita inestabilidad inicial (Goyal et al., 2017); coseno baja el LR suavemente sin saltos (Loshchilov \& Hutter, 2017, SGDR).
* **Problema observado con StepLR**: en la ejecución de referencia el LR se dividía por 10 cada 7 épocas → el modelo dejó de aprender pronto (infraajuste, macro-F1 0.36).
* **Colapso con LR alto**: SGD lr 1e-2 + Flatten de 50 176 entradas → la red predijo solo 2 clases (macro-F1 0.167). Lección: cambiar una cosa cada vez y ajustar el LR primero.
* **Label smoothing** (Szegedy et al., 2016), **mixup** (Zhang et al., 2018) y **pesos por clase 1/√frecuencia** (desbalance de clases) como regularizadores explorados en Fase 2.
* **Precisión mixta (AMP)**: \~2× más rápido en GPU sin pérdida de precisión relevante (Micikevicius et al., 2018).
* **Selección del mejor checkpoint por macro-F1 de validación** (la métrica de la competición), no por la pérdida.

## 4\. Búsqueda de hiperparámetros

* **Búsqueda aleatoria con garantía probabilística**: la probabilidad de que al menos una de n pruebas caiga en el mejor x % del espacio es 1 − (1 − x)ⁿ. Con 24 pruebas: \~71 % de dar con una del mejor 5 % y \~92 % con una del mejor 10 %. Supera a la rejilla cuando pocos hiperparámetros importan de verdad (Bergstra \& Bengio, 2012).
* **Poda (Successive Halving / Hyperband)**: repartir el presupuesto entre muchas pruebas y gastar más solo en las prometedoras (Li et al., 2017). Implementada como regla de parada por mediana (Golovin et al., 2017, Google Vizier): tras las épocas 4 y 8 se para el trial si va por debajo de la mediana.

  * Limitación observada: corta pronto a trials que arrancan lento (p. ej. SGD); un trial se podó por 0.001.
* **Búsqueda por fases (de lo grueso a lo fino)**: primero lo que más impacta (arquitectura, optimizador, LR), después regularización y secundarios alrededor del mejor. Limitación: es "voraz" — la Fase 2 hereda la arquitectura de la Fase 1.
* **Fases 1 y 2 sirven para elegir**: entrenamientos cortos (15 épocas, algunos podados) que solo comparan configuraciones; sus pesos se descartan.
* **Fase 3 produce el modelo final**: reentrena desde cero (no "sigue entrenando") las finalistas con 40 épocas y coseno estirado a 40 (el LR baja más despacio → la red aprende más). Reentrenar varias reduce el efecto del ruido entre ejecuciones cortas (no fiarse de un único entrenamiento). Usar el mismo train set no es redundante: es el único para aprender pesos.
* **Reparto en 2 máquinas sin sincronización**: lista de configuraciones generada con semilla fija (idéntica en ambas), reparto par/impar, un CSV por máquina. Diferencias pequeñas solo comparables dentro de la misma GPU.
* **Reproducibilidad**: semilla fija, cudnn determinista, entrenamiento final en una sola máquina.
* **Riesgo a mencionar**: la validación se usa para seleccionar → su F1 es algo optimista respecto al test.

### 4.1 Búsqueda profunda (1 máquina) — Fase A con diseño equilibrado

* **Espacio**: solo con los valores discretos hay **4608 combinaciones** (2·3·2·2·2·2·3·2·2·2·2), más LR y weight decay continuos. Todas las fases muestrean el mismo espacio; la Fase A no lo reduce.
* **Por qué no probarlas todas**: a ~7 min/prueba en la T4 → ~540 h (~22 días) frente a un máximo de 12 h por sesión de Kaggle.
* **Cómo se eligen**: no se buscan combinaciones concretas, sino que **cada valor de cada hiperparámetro aparezca el mismo nº de veces** (diseño equilibrado):
  1. Para cada hiperparámetro, lista de N posiciones con sus valores repetidos a partes iguales (p. ej. `input_size` 224 ×20 / 320 ×20; `base_channels` 32/48/64 ~13 veces cada uno con N = 40). LR y weight decay: Latin hypercube (un valor por tramo del rango).
  2. Cada lista se baraja por separado con semilla fija; la prueba *i* toma el elemento *i* de cada lista.
  3. Cada valor se prueba en contextos muy distintos → la media de sus pruebas indica qué valor funciona mejor en general (efecto principal; gráficos de la celda de análisis).
* **Tamaño N**: equilibrio cobertura/presupuesto. N = 40 → cada valor ≥ ~13 veces (no depende de una prueba con suerte). Limitación: no cubre bien las **interacciones** entre hiperparámetros → de eso se encarga la Fase B (evolutiva).
  * Ajuste previsto: en la T4 caben ~15–25 pruebas si la mitad son a 320 → N = 24 (cada valor ≥ 8 veces) deja tiempo para la Fase B.

## 5\. Decisiones que vienen de la teoría, no de la búsqueda

* Inicialización Kaiming, BatchNorm, convoluciones 3×3, coseno con warmup, AdamW, normalización de entrada → fijados por la literatura, no explorados (ahorra presupuesto).

## 6\. Alternativas consideradas (y por qué no)

* **Optimización bayesiana (TPE / procesos gaussianos)** (Bergstra et al., 2011): más eficiente, pero gana a la aleatoria solo tras \~10–20 pruebas y requiere sincronizar las 2 máquinas.
* **LR range test** (Smith, 2017): forma objetiva y barata de fijar el rango del LR. Posible mejora.
* **Population Based Training** (Jaderberg et al., 2017): requiere muchas pruebas en paralelo; excesivo para este presupuesto.

## 7\. Contexto del dataset (FireRisk)

* Shen et al., 2023 (DICTA): con el dataset completo (\~91k imágenes), ResNet-50 / ViT / DINO / MAE logran \~61–65 % de accuracy. Los errores se concentran entre clases vecinas (riesgo ordinal) → techo realista; nuestro subconjunto es mucho menor (3516 train).

\---

## Notas propias

* 

