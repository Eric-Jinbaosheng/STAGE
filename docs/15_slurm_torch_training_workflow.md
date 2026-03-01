# SLURM Torch Training Workflow

This is the end-to-end cluster workflow for:

- Python / Torch environment setup
- raw data pull
- processed data rebuild
- fine-tune training
- batch evaluation
- single-sample inference

All steps are provided as `sbatch` files under `slurm/`.

## Files

- `slurm/00_setup_env.sbatch`
- `slurm/01_pull_data.sbatch`
- `slurm/02_build_processed_data.sbatch`
- `slurm/03_train_finetune.sbatch`
- `slurm/04_eval_batch.sbatch`
- `slurm/05_single_infer.sbatch`

## 1. Environment Setup

Submit:

```bash
sbatch slurm/00_setup_env.sbatch
```

Useful overrides:

```bash
sbatch --export=ALL,WORKDIR=/path/to/NLP_Final_Project,CONDA_ENV_DIR=/scratch/$USER/NLP_Final_Project/.conda_torch,PYTHON_MODULE=anaconda3/2025.06,USE_CUDA_MODULE=0 slurm/00_setup_env.sbatch
```

This creates a conda environment on scratch and installs:

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `pyarrow`
- `h5py`
- `huggingface_hub`

By default it does not install:

- `torchaudio`
- `robosuite`

If you want the extra simulation dependencies too:

```bash
sbatch --export=ALL,INSTALL_EXTRA_SIM_DEPS=1 slurm/00_setup_env.sbatch
```

Current default is already `INSTALL_EXTRA_SIM_DEPS=1`, which installs the lighter LIBERO data-pull stack:

- `gymnasium`
- `termcolor`
- `psutil`
- `tensorboard`
- `tensorboardX`
- `imageio`
- `imageio-ffmpeg`
- `matplotlib`
- `robomimic --no-deps`

This avoids the heavier `egl_probe` build path while still allowing `third_party/LIBERO/libero/lifelong/datasets.py` to run.

This script is already adapted for the current `torch` cluster behavior:

- uses `anaconda3/2025.06`
- forces Python 3.11 inside the conda env
- moves conda and pip writes to `/scratch`
- avoids the home-directory quota issue
- leaves the `ROBOMIMIC WARNING: No private macro file found!` warning untouched because it is non-fatal for dataset download

## 2. Pull Data

Submit:

```bash
sbatch slurm/01_pull_data.sbatch
```

If you already copied raw LIBERO and handover files onto the cluster, disable downloading:

```bash
sbatch --export=ALL,DOWNLOAD_LIBERO=0 slurm/01_pull_data.sbatch
```

If you have DexYCB cache available and want to export handover raw episodes:

```bash
sbatch --export=ALL,DEXYCB_CACHE_DIR=/path/to/dex-ycb-cache slurm/01_pull_data.sbatch
```

Notes:

- LIBERO download includes only:
  - `libero_spatial`
  - `libero_object`
  - `libero_goal`
- It does not download:
  - `libero_10`
  - `libero_90`
  - `libero_100`

## 3. Rebuild Processed Data

Submit:

```bash
sbatch slurm/02_build_processed_data.sbatch
```

This rebuilds:

- `data/processed/libero_index.parquet`
- `data/processed/handover_index.parquet`
- `data/processed/schema_labels.parquet`
- `data/processed/object_vocab.json`
- `data/processed/meta.json`

## 4. Fine-Tune Training

Submit:

```bash
sbatch slurm/03_train_finetune.sbatch
```

Typical override:

```bash
sbatch --export=ALL,OUT_DIR=artifacts/finetune_baseline,STEPS=5000,BATCH_SIZE=64,FRAME_WINDOW=3,VIEWS=agentview_rgb,eye_in_hand_rgb slurm/03_train_finetune.sbatch
```

If you want to reuse an existing text vocab:

```bash
sbatch --export=ALL,TEXT_VOCAB_PATH=artifacts/finetune_baseline/text_vocab.json slurm/03_train_finetune.sbatch
```

Key outputs:

- `checkpoint.pt`
- `metrics.json`
- `text_vocab.json` (only when not reusing an existing vocab file)

## 5. Batch Evaluation

Submit:

```bash
sbatch slurm/04_eval_batch.sbatch
```

Typical override:

```bash
sbatch --export=ALL,CHECKPOINT=artifacts/finetune_baseline/checkpoint.pt,TEXT_VOCAB=artifacts/finetune_baseline/text_vocab.json,MAX_SAMPLES=512,BATCH_SIZE=32,OUT_PATH=artifacts/batch_eval/result.json slurm/04_eval_batch.sbatch
```

The output JSON contains:

- overall accuracy
- confusion matrices
- per-class precision / recall / support
- a short prediction preview

## 6. Single-Sample Inference

Submit:

```bash
sbatch slurm/05_single_infer.sbatch
```

Typical override:

```bash
sbatch --export=ALL,CHECKPOINT=artifacts/finetune_baseline/checkpoint.pt,TEXT_VOCAB=artifacts/finetune_baseline/text_vocab.json,EPISODE_ID=open_the_middle_drawer_of_the_cabinet_demo:demo_0,FRAME_ID=0,OUT_PATH=artifacts/single_infer/result.json slurm/05_single_infer.sbatch
```

## Recommended Order

Run in this order:

1. `sbatch slurm/00_setup_env.sbatch`
2. `sbatch slurm/01_pull_data.sbatch`
3. `sbatch slurm/02_build_processed_data.sbatch`
4. `sbatch slurm/03_train_finetune.sbatch`
5. `sbatch slurm/04_eval_batch.sbatch`
6. `sbatch slurm/05_single_infer.sbatch`

## Important Cluster Assumptions

- These scripts assume your SLURM cluster accepts `--partition=cpu` and `--partition=gpu`
- If your cluster uses different partition names, edit the `#SBATCH` lines
- If your cluster requires `--account`, add it to the `#SBATCH` lines
- If your cluster manages modules differently, adjust:
  - `PYTHON_MODULE`
  - `CUDA_MODULE`
- If CUDA version differs, change `TORCH_INDEX_URL` in `slurm/00_setup_env.sbatch`
- `cuda_available=False` on the login node can still be normal; verify GPU access on a GPU compute job

These scripts are designed to be editable templates, not hard-coded to one cluster.
