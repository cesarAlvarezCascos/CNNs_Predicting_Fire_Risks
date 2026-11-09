# Fire risk classification from satellite images

Audio Processing, Video Processing and Computer Vision  · UC3M · 2026/27

Classification of satellite images into 7 fire-risk classes with CNNs: a custom network trained from scratch and a fine-tuned pretrained network. Everything lives in `Lab_4_AVCV_.ipynb`.

> **Just want to run it?** See [HOW_TO_RUN.md](HOW_TO_RUN.md): the self-contained `Lab_4_AVCV_NoDependencies.ipynb` only needs `db.zip`.

## Dataset

A reduced subset (5,861 RGB images, 320×320) of the [FireRisk](https://github.com/CharmonyShen/FireRisk) dataset: Shen, S., Seneviratne, S., Wanyan, X., & Kirley, M. (2023). *FireRisk: A remote sensing dataset for fire risk assessment with benchmarks using supervised and self-supervised learning.* DICTA 2023, pp. 189-196. IEEE.

Download the **release asset** `db.zip` (Releases section of this repository) and place it **as a zip** in the same folder as the notebook. Do not unzip it: the notebook extracts it automatically.

## Final models (our experiments)

With our search on this dataset, the models chosen for the final Codabench submission were:

| Category | Model | Public Codabench macro-F1 |
|---|---|---|
| CUSTOM (trained from scratch) | `n07_60ep`: 4 blocks of 2 × [Conv 3×3 → BatchNorm → ReLU] + MaxPool, 48–384 filters, global average pooling, 60 epochs | 0.50 |
| FINE-TUNING (ImageNet-pretrained) | `swin_p2_1`: Swin-T with a new 7-class output layer, 30 epochs | 0.49 |

Both were trained on 224×224 random crops and are evaluated on the full 320×320 image. These names are results of *our* runs, not something the code depends on: the notebook reads them from the config files below, so the same pipeline can search, select and train other models (or on another dataset).

Their weights (`.pt` files) are too large for git and are published in the **`models` release** of this repository. Download them and put them in `final/` (or in the paths listed in the config files). **Keep the file names unchanged**: the notebook looks for them by name.

## Directory structure

Only `Lab_4_AVCV_.ipynb`, `README.md` and `final/` are versioned in git. The other folders are created by the notebook when it runs, or are downloaded separately.

```
LAB_4_CNNs/
├── Lab_4_AVCV_.ipynb   the notebook: data, both models, search, training, evaluation, Codabench files
├── db.zip              the dataset (Releases; the notebook extracts it to data/)
├── final/              INPUT: which model to use and how to rebuild it, plus what was submitted
├── outputs/            OUTPUT of a run: weights, test predictions, run records, submission ZIP
└── search/             OUTPUT of the hyperparameter search: one CSV row per trial, learning curves
```

| Folder | Written by | Main files |
|---|---|---|
| `final/` | you (the notebook only reads it) | `custom_final_config.json`, `ft_final_config.json` (see next section); `output_*_submitted.csv` (the predictions actually submitted, used to check reproducibility); `*_manifest.json` (training / run records: settings, versions, scores, SHA-256 hashes of weights and predictions) |
| `outputs/` | every run | `deep_search/<trial>.pt` and `Lab_4_OUT/ft/<trial>.pt` (weights); `Lab_4_OUT/output_custom.csv`, `output_ft.csv` (test predictions); `Lab_4_OUT/*_run_manifest.json` (record of the run); `Lab_4_OUT/codabench_submission.zip` |
| `search/` | only when `RUN_SEARCH = 1` | `results_p<phase>_m<machine>.csv` (one row per trial: settings, best validation macro-F1, per-epoch scores); `*_epochs.csv` (learning curves); `overnight_p5/` (phase-5 runs and determinism test); `ft/` (the same for fine-tuning) |

A **hash** (SHA-256) is a file fingerprint: if one byte changes, the hash changes. The notebook compares hashes and probabilities to confirm it reproduces the submitted predictions (on a different GPU the values match to ~1e-6 but the hash may not).

The search runs in **phases** (random trials with early pruning, then longer runs around the best ones). Phases 1–2 can be split between two computers (`MACHINE_ID`, `N_MACHINES`, hence `m0` / `m1` in the file names). By default the notebook only *displays* the stored results (`RUN_SEARCH = 0`).

## Running the notebook: the `*_final_config.json` files

The two config files in `final/` are the **input** that decides what the notebook trains or loads, so you change how it runs without editing any cell. `custom_final_config.json` drives section 3.5 (custom model) and `ft_final_config.json` drives section 4.4 (fine-tuning).

```json
{"selected": "<candidate id>",
 "candidates": {"<candidate id>": {
     "arch":    {...},   // architecture (channels, blocks, head, dropout, train_crop; FT: network, drop_path)
                         //   eval_size: validation crop during training (selects the best epoch)
                         //   predict_size: crop used for the final val report and the test predictions
     "train":   {...},   // optimiser, lr, weight decay, batch, warm-up, loss options, augmentation, epochs, seed
     "runtime": {...},   // num_workers, strict_determinism, pool_impl, preload, amp (how the original run was executed)
     "reproduce": {"mode": "retrain" | "checkpoint", "checkpoint": "<path to .pt>",
                   "submitted_csv_sha256": "<hash or null>"},
     "provenance": {...} // informational: where the candidate came from, its validation score, Codabench score
     // FT only: "members": [ids]  -> ensemble = mean of the members' probabilities
 }}}
```

What you can change:

| Input | Effect |
|---|---|
| `selected` | Which candidate is the final model. `null` → the notebook stops at 3.5 / 4.4 with a clear message. |
| `reproduce.mode` | `"checkpoint"`: load the `.pt` and only predict (exact reproduction of a submission, minutes). `"retrain"`: train from scratch with `arch` + `train` + `runtime` (full reproduction, hours). |
| `reproduce.checkpoint` | Where the weights are read from (checkpoint mode) or compared against (retrain mode). |
| `arch`, `train` | The hyperparameters of the model to train (`eval_size` = validation crop while training, `predict_size` = crop of the final predictions; ours: 224 and 320). Add a new candidate with any configuration (for example the best trial of a new search, copied from its `cfg` column in `search/results_*.csv`) and select it. |
| `runtime` | Reproducibility settings of the original run. Exact retraining needs the same ones (and the same GPU type); the notebook warns if the current machine differs (e.g. Windows forces `num_workers = 0`). |
| `submitted_csv_sha256` | If set, the new predictions are compared with `final/output_*_submitted.csv`. `null` skips the check. |

Two switches in the first code cell complete the picture: `RUN_SEARCH = 1` re-runs the hyperparameter search (writes `search/` and new weights), and `SMOKE_TEST = 1` uses 64 training images and one epoch, only to check that everything executes.

Typical uses:
- **Reproduce our submission**: keep the files as they are (`mode: "checkpoint"`) and run all cells.
- **Retrain the known best model** without searching: `RUN_SEARCH = 0`, `mode: "retrain"`.
- **Start from scratch / another dataset**: `RUN_SEARCH = 1`, then add the winning configuration as a candidate, set `selected`, and run again with `mode: "retrain"`.
