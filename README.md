# Fire risk classification from satellite images

Audio Processing, Video Processing and Computer Vision  · UC3M · 2026/27

Classification of satellite images into 7 fire-risk classes with CNNs: a custom network trained from scratch and a fine-tuned pretrained network. Everything lives in `Lab_4_AVCV_.ipynb`.

## Dataset

A reduced subset (5,861 RGB images, 320×320) of the [FireRisk](https://github.com/CharmonyShen/FireRisk) dataset: Shen, S., Seneviratne, S., Wanyan, X., & Kirley, M. (2023). *FireRisk: A remote sensing dataset for fire risk assessment with benchmarks using supervised and self-supervised learning.* DICTA 2023, pp. 189-196. IEEE.

Download the **release asset** `db.zip` (Releases section of this repository) and place it **as a zip** in the same folder as the notebook. Do not unzip it: the notebook extracts it automatically.

## Final models

The two models used for the final Codabench submission are:

| Category | Model | Public Codabench macro-F1 |
|---|---|---|
| CUSTOM (trained from scratch) | `n07_60ep`: 4 blocks of 2 × [Conv 3×3 → BatchNorm → ReLU] + MaxPool, 48–384 filters, global average pooling, 60 epochs | 0.50 |
| FINE-TUNING (ImageNet-pretrained) | `swin_p2_1`: Swin-T with a new 7-class output layer, 30 epochs | 0.49 |

Both were trained on 224×224 random crops and are evaluated on the full 320×320 image.

Their weights (`.pt` files) are too large for git and are published in the **`models` release** of this repository. Download them and put them in `final/` (or in the paths listed below). **Keep the file names unchanged**: the notebook looks for them by name.

## Directory structure

Only `Lab_4_AVCV_.ipynb`, `README.md` and `final/` are versioned in git. The other folders are created by the notebook when it runs, or are downloaded separately.

```
LAB_4_CNNs/
├── Lab_4_AVCV_.ipynb      the notebook: data, both models, training, evaluation and the Codabench files
├── db.zip                 the dataset (download from the Releases; the notebook extracts it to data/)
├── final/                 which models are used for the final submission, and proof of what was submitted
├── outputs/               files the notebook writes: model weights, test predictions, submission archive
└── search/                results of the hyperparameter search (one CSV row per trial)
```

### `final/`: models used for the submission

Read by the notebook to know **which model to load and how to rebuild it**, and to check that it reproduces the submitted predictions.

| File | What it contains |
|---|---|
| `custom_final_config.json` | Candidate custom models with their full architecture and training settings; `selected` names the one the notebook uses (`n07_60ep`). Also whether to load its weights or retrain it, and where each result came from |
| `ft_final_config.json` | Same for the fine-tuned models (`selected`: `swin_p2_1`) |
| `custom_final_manifest.json` | Training record of the first submitted custom model (`p4_09_B_wide48x2`): settings, per-epoch scores, software versions and file fingerprints (SHA-256 hashes) |
| `ft_final_manifest.json` | Training record of `swin_p2_1`: pretrained weights used, settings, best epoch, validation score and hashes |
| `custom_run_manifest.json` | Record of the final notebook run for the custom model: settings, environment (GPU, library versions), validation score, and the hashes of the weights and predictions it used and produced |
| `ft_run_manifest.json` | Same for the fine-tuned model |
| `output_custom_submitted.csv` | Exact test predictions submitted to Codabench for the custom category (1,173 rows × 7 class probabilities) |
| `output_ft_submitted.csv` | Exact test predictions submitted for the fine-tuning category |

A **hash** (SHA-256) is a fingerprint of a file: if a single byte changes, the hash changes. The notebook compares hashes and probabilities to confirm that it produces the same predictions that were submitted.

### `outputs/`: files written by the notebook (not versioned)

| Path | What it contains |
|---|---|
| `outputs/deep_search/` | Weights of the custom models trained in search phases 3–5, one `<trial>.pt` per model. The final one is `n07_60ep.pt`; `p4_09_B_wide48x2.pt` is the earlier candidate |
| `outputs/Lab_4_OUT/output_custom.csv` | Test predictions of the custom model produced by the last run |
| `outputs/Lab_4_OUT/output_ft.csv` | Test predictions of the fine-tuned model produced by the last run |
| `outputs/Lab_4_OUT/codabench_submission.zip` | The two CSV files above, packed as Codabench expects |
| `outputs/Lab_4_OUT/*_run_manifest.json` | Record of the last run (copied to `final/` for the submitted run) |
| `outputs/Lab_4_OUT/ft/` | Weights of the six fine-tuned models trained in phase 2 of the fine-tuning search (`cnx_` ConvNeXt-T, `r50_` ResNet-50, `swin_` Swin-T). The final one is `swin_p2_1.pt` |

### `search/`: hyperparameter search results (not versioned)

The hyperparameters were chosen with a **random search**: each **trial** trains the network with a randomly drawn combination of settings, then compares trials by validation macro-F1. Weak trials were stopped early (*pruning*). The search ran in **phases**, each starting from the best result of the previous one. Phases 1 and 2 were split between two computers, **machine 0** (`m0`) and **machine 1** (`m1`); phases 3–5 ran on machine 1 only. The notebook only displays these results: the search is not re-run (`RUN_SEARCH = 0`).

| File | What it contains |
|---|---|
| `results_p1_m0.csv`, `results_p1_m1.csv` | Phase 1 (architecture, optimiser and learning rate), 24 trials of 15 epochs: the even-numbered trials ran on machine 0, the odd-numbered ones on machine 1. One row per trial: settings, best validation macro-F1 and its epoch, and the validation score of every epoch |
| `results_p2_m0.csv`, `results_p2_m1.csv` | Phase 2 (regularisation and secondary settings around the best phase-1 trial), 16 trials split the same way |
| `results_p3_m1.csv` | Phase 3: the three best phase-2 configurations retrained for 40 epochs |
| `results_p4_09_m1.csv` | Phase 4: one change at a time on the best configuration (class weights; wider network) |
| `results_p5_m1.csv` | Phase 5: twelve runs on Kaggle with fully deterministic training (more epochs, wider or deeper networks, other seeds, larger crops) |
| `p3_*_epochs.csv`, `p4_*_epochs.csv` | Training and validation loss and macro-F1 at every epoch of one phase-3 or phase-4 run (learning curves) |
| `p3_epochs_f1.csv` | The phase-3 learning curves of the three finalists side by side |
| `overnight_p5/<run>_epochs.csv` | Learning curves of each phase-5 run (`n07_60ep_epochs.csv` belongs to the final custom model) |
| `overnight_p5/overnight_summary.csv` | One-line summary per phase-5 run: mean of the last 10 epochs, best score, train score, final validation loss |
| `overnight_p5/determinism_check.json` | Result of training the same model twice and checking that the weights are identical (reproducibility test) |
| `ft/ft_results.csv` | Fine-tuning search: one row per trial (network, settings, best and last-5-epoch validation macro-F1) |
| `ft/ft_leaderboard_phase1.csv`, `ft/ft_leaderboard_phase2.csv` | Fine-tuning trials ranked by validation macro-F1: phase 1 (10 random configurations per network, 10 epochs) and phase 2 (the 2 best of each network, 30 epochs) |
| `ft/ft_per_class_f1.csv` | Validation F1 of each class for the six phase-2 fine-tuned models |
| `ft/<trial>_epochs.csv` | Learning curves of each fine-tuning trial |
| `ft/determinism_check.json` | Reproducibility test of each pretrained network, plus its training time per epoch |
