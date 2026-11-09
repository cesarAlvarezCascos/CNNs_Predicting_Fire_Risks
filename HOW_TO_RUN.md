# How to run `Lab_4_AVCV_NoDependencies.ipynb`

## What you need

- The notebook `Lab_4_AVCV_NoDependencies.ipynb`.
- The dataset `db.zip` next to the notebook (on Kaggle: *Add Input*; the notebook finds and extracts it).
- A GPU. For the same results as our submission: **Kaggle, GPU T4**, with **Internet ON** (downloads the ImageNet weights of Swin-T).

Nothing else: no `final/`, `search/` or `outputs/` files are needed. Run all cells top to bottom.

## Options (first code cell)

| Variable | Default | Effect |
|---|---|---|
| `USE_CHECKPOINTS` | `False` | `False`: trains both final models from scratch (~1 h on a T4). `True`: loads the trained `.pt` files and only predicts (minutes) |
| `CHECKPOINT_DIR` | `.` | Folder with `n07_60ep.pt` and `ft_final_swin_p2_1.pt` (only with `USE_CHECKPOINTS = True`; on Kaggle the inputs are also searched) |
| `RUN_SEARCH` | `0` | `1` re-runs the whole hyperparameter search (**many hours**). Not needed: the final configurations are already in the notebook |
| `SMOKE_TEST` | `0` | `1` uses 64 images and 1 epoch, only to check that everything executes |
| `PREDICT_SIZE`, `FT_PREDICT_SIZE` | `320` | Image size of the final predictions (320 = full image) |

> **Recommended:** if you have the trained models (`n07_60ep.pt` and `ft_final_swin_p2_1.pt`), set `USE_CHECKPOINTS = True`. It saves about 1 h of training. It also keeps the results exactly reproducible, because the loaded weights are the same ones we used, with every decimal intact. Retraining on the GPU can change the weights very slightly.

The result is `outputs/Lab_4_OUT/codabench_submission.zip` (`output_custom.csv` + `output_ft.csv`). At the end, the notebook compares its predictions with the fingerprint (SHA-256) of the files we submitted to Codabench.

## How the original notebook works, and what changes here

The main notebook (`Lab_4_AVCV_.ipynb`) reads the final models from `final/custom_final_config.json` and `final/ft_final_config.json`. These files list the candidate models and their settings, and control which one is used and whether it is retrained or loaded. With the search enabled, it also writes the search results (one row per trial with its configuration, leaderboards, and per-epoch training/validation curves) to `search/`, and the weights, predictions and run records to `outputs/`.

This **NoDependencies** version does not read any of those files. The settings of the two selected models (`n07_60ep` and `swin_p2_1`) are written directly in the notebook (`ENTRY` in Part 1, `FT_ENTRY` in section 4.4). The search sections only show their results when the `search/` folder is present.
