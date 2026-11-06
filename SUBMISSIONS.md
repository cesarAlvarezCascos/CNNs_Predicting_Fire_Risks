# Envíos a CodaBench y experimentos clave

Competición CodaBench 18327 · Métrica: macro-F1 · Límite: 2 envíos al día por equipo (`100475161_100475191`) · Fin: 9 nov 2026, 23:59.

## Registro de envíos

| Fecha | ID | Custom (modelo) | Val macro-F1 | Público custom | FT (modelo) | Público FT | Quién | Notas |
|---|---|---|---|---|---|---|---|---|
| 2026-11-05 19:36 | 1036062 | `p4_09_B_wide48x2` | 0.4956 (ép. 36/40) | **0.49** (9.º) | **Placeholder**: copia de la custom | 0.49 | Dani | El `output_ft.csv` NO es un fine-tuning; sustituir por uno real |

Al añadir un envío: anotar el checkpoint, la configuración (o el `trial_id`) y el SHA-256 de los CSV enviados.

## Modelo custom ganador: `p4_09_B_wide48x2`

- **Checkpoint:** `MyDrive/L4/final/custom_final.pt` (copia de `MyDrive/L4/outputs/Lab_4_2026_2027/search/p4_09_B_wide48x2.pt`)
  - SHA-256 `829214a456a3e149d58089fbb5937bf68f3a53a0ed4f0910ee6f7b33f7e272bd`
- **Predicciones enviadas:** `final/output_custom_submitted.csv`
  - SHA-256 `b706a460d879f4c0f4c7eff7057becc46c3d95876e640bca7b7e5986a1affeb4`
  - Recargar el checkpoint reproduce estas probabilidades (máx. |dif| 5e-10, mismas clases).
- **Manifiesto completo:** `final/custom_final_manifest.json` (configuración, historial por época, versiones y entorno).

| Grupo | Valor |
|---|---|
| Arquitectura | `CustomNet`, 4 bloques, filtros 48-96-192-384, **2 × [Conv3×3 (padding 1) → BN → ReLU]** + MaxPool 2×2 por bloque; **GAP** → FC 256 → ReLU → Dropout 0.0 → FC 7; inicialización Kaiming; **2.739.447 parámetros** |
| Optimizador | AdamW, lr 0.000517, weight decay 0.0753 |
| Scheduler | 1 época de warm-up lineal y luego cosine hasta 0 (`LambdaLR` por época), **40 épocas** |
| Loss | CrossEntropy con pesos `sqrt_inv` (1/√n, media 1), sin label smoothing |
| Datos | batch 128; train `RandomCrop(224)` → `GPUAugment` preset *light* (γ 0.1, brillo/contraste/saturación 0.05; flips y rot90); val/test `CenterCrop(224)` + Normalize ImageNet; sin mixup |
| Ejecución | SEED 42, AMP fp16, `cudnn.deterministic=True`, `benchmark=False`, **`NUM_WORKERS=0`**, train precargado en RAM; se guarda la época con mejor val macro-F1 |
| Entorno | Colab, Tesla T4, Python 3.13.16, torch 2.11.0+cu130, torchvision 0.26.0+cu130, CUDA 13.0, cuDNN 92700, numpy 2.4.6, pandas 2.2.3 |

Reentrenar desde cero en GPU no es idéntico bit a bit (`AdaptiveAvgPool2d` no es determinista en CUDA, AMP). El envío se reproduce exactamente cargando el checkpoint.

## Resumen de experimentos (val macro-F1)

La "media de las 10 últimas épocas" es la métrica robusta. El máximo de una sola época tiene ruido de ±0.01–0.02.

| Fase | Experimento | Cambio | Media 10 últ. | Máx. | Train 10 últ. | Conclusión |
|---|---|---|---|---|---|---|
| 3 | `p3_p2_07` | finalista (32 filtros, 1 conv, GAP, dropout 0.3, aug strong) | 0.456 | 0.464 | 0.450 | techo ~0.45, train ≈ val → underfitting |
| 3 | `p3_p2_09` | finalista (dropout 0, aug light) | 0.453 | 0.464 | 0.492 | ídem |
| 3 | `p3_p2_11` | finalista (mixup 0.2, smoothing 0.1) | 0.447 | 0.466 | 0.297* | ídem (*el F1 de train no tiene sentido con mixup) |
| 4 | `p4_09_A_inv` | p2_09 + `class_weights='inv'` | 0.431 | 0.450 | 0.456 | **empeora (−0.022)** |
| 4 | `p4_A_inv` | p2_07 + `class_weights='inv'` | 0.419 | 0.437 | 0.413 | **empeora (−0.037)**, confirma lo anterior |
| 4 | **`p4_09_B_wide48x2`** | p2_09 + 48 filtros y 2 convs por bloque | **0.483** | **0.496** | 0.525 | **mejora (+0.030)** |
| 4 | `p4_09_D_wide64x2` | p2_09 + 64 filtros y 2 convs por bloque | — | — | — | interrumpido en la época 5 |
| 4 (Kaggle) | `p4_A_inv` | p2_07 + `inv` (repetición en Kaggle T4) | 0.421 | 0.436 | 0.412 | empeora (−0.036). Repetido en Colab daba 0.419: **misma config en otra GPU → ±0.002** |
| 4 (Kaggle) | `p4_B_wide48x2` | p2_07 + 48 filtros y 2 convs por bloque | 0.480 | 0.488 | 0.495 | **mejora (+0.024)**: confirma B en la segunda base |
| 4 (Kaggle) | `p4_C_wide_noreg` | p2_07 + 48×2, dropout 0, aug light | 0.482 | 0.493 | 0.541 | mejora (+0.026); prácticamente igual que `p4_09_B` (solo cambian lr/wd) |

Conclusión de la fase 4: **la capacidad es lo que importa.** Las tres variantes anchas (`p4_09_B`, `p4_B`, `p4_C`) quedan entre 0.480 y 0.483 en la media de las 10 últimas épocas, aunque difieren en lr, weight decay, dropout y augmentation. Los pesos `inv` empeoran en las dos bases.

Notebook de Kaggle: `experiments/Lab_4_p4_kaggle.ipynb` (Tesla T4, `preload=False`, 4 workers).

## Fase 5 · Ejecución nocturna en Kaggle, 100 % determinista (6 nov 2026)

Notebook: `experiments/Lab_4_overnight_kaggle.ipynb` (la versión exacta que se ejecutó, descargada con `kaggle kernels pull`).
- Configuración común: Tesla T4, `torch.use_deterministic_algorithms(True)` estricto, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, cuDNN determinista sin benchmark, GAP determinista (`x.mean`), semilla 42, `NUM_WORKERS=4`, `preload=False`, AMP fp16.
- **Prueba de determinismo superada:** dos entrenamientos iguales dan pesos idénticos (diferencia máxima 0.0; ver `search/overnight_p5/determinism_check.json`).
- 12 experimentos en 5,4 h, todos en modo estricto, sin errores ni falta de memoria.

Bases: **W09** = `p4_09_B_wide48x2` y **W07** = `p4_B_wide48x2` (48 filtros, 2 convs por bloque). Para comparar, la referencia válida es la del **mismo entorno y la misma semilla**: `n09_ref` = 0.472.

| Experimento | Cambio | Media 10 últ. | Máx. (época) | Train 10 últ. | Val loss final | Params |
|---|---|---|---|---|---|---|
| **`n09_60ep`** | W09, 60 épocas | **0.489** | **0.5005** (48) | 0.578 | 1.160 | 2,74 M |
| **`n07_60ep`** | W07, 60 épocas | **0.489** | **0.5024** (36) | 0.524 | **1.141** | 2,74 M |
| `n09_crop288_eval320` | W09, recortes de 288 y evaluación a 320 | 0.483 | 0.489 | 0.535 | 1.159 | 2,74 M |
| `n09_combo` | W09, 64×2 + 60 ép. + 288/320 | 0.482 | 0.490 | 0.561 | 1.171 | 4,82 M |
| `n07_crop288_eval320` | W07, recortes de 288 y evaluación a 320 | 0.480 | 0.491 | 0.501 | 1.150 | 2,74 M |
| `n07_5blocks` | W07, 5 bloques | 0.476 | 0.482 | 0.474 | 1.172 | 6,90 M |
| `n07_64x2` | W07, 64 filtros | 0.473 | 0.479 | 0.474 | 1.177 | 4,82 M |
| `n09_ref` | W09 tal cual (referencia) | 0.472 | 0.488 | 0.534 | 1.164 | 2,74 M |
| `n09_64x2` | W09, 64 filtros | 0.467 | 0.479 | 0.504 | 1.211 | 4,82 M |
| `n09_5blocks` | W09, 5 bloques | 0.462 | 0.471 | 0.492 | 1.200 | 6,90 M |
| `n09_seed43` | W09, semilla 43 | 0.459 | 0.471 | 0.521 | 1.190 | 2,74 M |
| `n09_seed44` | W09, semilla 44 | 0.456 | 0.466 | 0.510 | 1.203 | 2,74 M |

Conclusiones:
1. **Ruido por semilla grande:** W09 con semillas 42/43/44 → 0.472 / 0.459 / 0.456 (media 0.463, desviación ≈ 0.008). Con una sola semilla, diferencias < ~0.015 no son concluyentes. El 0.483 de W09 en Colab fue una semilla favorable.
2. **60 épocas es lo único que mejora con claridad en las dos bases:** media de las 10 últimas 0.489 en ambas (+0.017 frente a `n09_ref` con la misma semilla) y máximos > 0.50.
3. **Más anchura (64) o más profundidad (5 bloques) no ayuda;** la combinación tampoco supera a 60 épocas solas.
4. **Recortes de 288 + evaluación a 320:** como mucho +0.01 respecto a la referencia, dentro del ruido.

Candidato a modelo custom final: **`n07_60ep`** (máx. 0.5024; menor val loss y menor diferencia train-val que `n09_60ep`). Es reproducible exactamente con el notebook de esta fase.

Archivos: `search/results_p*_m*.csv` (todas las pruebas), `search/*_epochs.csv` (F1 y loss de train/val por época), `search/p3_epochs_f1.csv` (tabla de la fase 3).
