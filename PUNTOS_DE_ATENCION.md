# LAB 4 – Puntos de atención

Marca `\[x]` al resolver cada punto y anota la decisión tomada. Última actualización: 2026-11-02.
**Deadline: 9 de noviembre, 23:59.**

## 1\. Ambigüedades (por aclarar / decidir)

* \[ ] **Métrica oficial de Codabench**: el notebook solo habla de macro-F1 en validación. Comprobar en la guía de Codabench (aún sin leer).
* \[ ] **Combinación de las dos categorías** (CUSTOM + FT) en la nota, baseline y mapeo lineal: no especificado. Preguntar al profesor o mirar la guía.
* \[ ] **¿Entrenar el modelo final con train+val?** No está dicho. Por defecto: NO (ver sección 2, reproducibilidad).
* \[ ] **¿Se permite ensemble o TTA?** No se menciona. Preguntar antes de usarlo.
* \[ ] **Límite de "arquitectura externa" en CUSTOM**: ¿vale escribir a mano bloques tipo ResNet/SE/CBAM? Preguntar.
* \[ ] **Reparto de contribuciones** en la pareja: definirlo pronto y reflejarlo en el informe.

## 2\. Puntos técnicos del notebook

* \[ ] **Rutas**: `DATA\_ROOT='data'` extrae a `data/db`; yo descomprimí a mano en `db/`. Ajustar el notebook para que se encargue él.
* \[ ] **`MAX\_TRAIN\_SAMPLES=500`** por defecto: poner **0** para el modelo final (el notebook exige el train completo).
* \[ ] **StepLR(step\_size=7) con 5 épocas**: el LR nunca baja. Cambiar scheduler o épocas.
* \[ ] **`CUSTOM\_LR=1e-2` con SGD**: puede ser inestable en la red custom. Probar valores en validación.
* \[ ] **Entrada 224 sobre imágenes 320×320**: el `CenterCrop` descarta los bordes. Decidir y justificar (recortar vs. redimensionar).
* \[ ] **Aumentos**: flips y rotaciones de 90° son plausibles en satélite; cuidado con cambios de color que alteren la clase. Validar cada uno con val.
* \[ ] **Desbalance** (Water 86 en train frente a Very Low 1088): macro-F1 pesa igual todas las clases. Valorar pesos en la pérdida o muestreo; `train\_model` exige `CrossEntropyLoss(reduction='mean')`.
* \[ ] **Fine-tuning**: usar el preprocesado (tamaño, media/std) propio de los pesos del modelo elegido.
* \[ ] **CUSTOM sin preentrenado ni arquitecturas externas**; arquitectura completa visible en celdas del notebook.

## 3\. Reproducibilidad y entrega

* \[ ] Fijar semillas y guardar la **configuración final exacta** (épocas, LR, aumentos, versión de librerías). Se audita el código; si no se reproduce, descalificación.
* \[ ] El modelo final debe poder regenerarse con un número fijo de épocas y selección de checkpoint por val (por eso train+val complica las cosas).
* \[ ] CSV de salida: **1173×7**, sin cabecera ni ID, mismo orden que `test\_reduced.csv`, columnas 0–6, filas que suman 1.
* \[ ] Nombres exactos dentro del ZIP de Codabench: `output\_custom.csv` y `output\_ft.csv`.
* \[ ] **Dos entregas distintas**: `codabench\_submission.zip` (Codabench) y un `.zip` en Aula Global con informe + notebook.
* \[ ] Informe: 3 páginas + 1 de tablas/figuras/referencias; experimentos con su objetivo; contribución de cada miembro.

## 4\. Entorno

* \[ ] El portátil tiene una **RTX 4050 (6 GB)**, pero el PyTorch instalado es la versión **CPU** (`2.6.0+cpu`): reinstalar la build con CUDA para usarla. Alternativa: laboratorio virtual / Colab.
* \[ ] Con 6 GB de VRAM, vigilar el `batch_size` (64 puede no caber en modelos grandes).
* \[ ] Copiar `db.zip` al disco local de la máquina GPU antes de entrenar (no leer imágenes desde una unidad de red).
* \[ ] Guardar checkpoints y resultados fuera del disco temporal de la sesión.
* \[ ] Pandas falló en mi modo aislado por `dateutil` faltante: verificar el entorno completo.

### Reglas del profesor para GPU (pixel.tsc.uc3m.es / Colab)

* \[ ] Máquinas del lab 4.0B01A (https://pixel.tsc.uc3m.es/): disponibles ~18:00–02:00 (se apagan a las 2:00), fines de semana siempre.
* \[ ] Ubuntu permite `num_workers>0`; en **Windows, `num_workers=0` obligatorio**.
* \[ ] Primera celda del notebook: `!pip3 install torch torchvision torchaudio` (+ pandas, scikit-image, scikit-learn si faltan).
* \[ ] **Al apagarse la máquina se borra todo**: guardar checkpoints y outputs en Google Drive a medida que se entrena.
* \[ ] Copiar los archivos a un directorio local de la máquina y lanzar `jupyter-notebook` desde ahí.

## 5\. Decisiones tomadas

|Fecha|Decisión|Motivo|
|-|-|-|
||||



