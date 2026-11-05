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

Archivos: `search/results_p*_m*.csv` (todas las pruebas), `search/*_epochs.csv` (F1 y loss de train/val por época), `search/p3_epochs_f1.csv` (tabla de la fase 3).
