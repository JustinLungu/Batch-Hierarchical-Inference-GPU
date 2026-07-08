# Shell Scripts

This folder contains shell helpers for preparing local experiments:

- `setup_env.sh` syncs the uv environment from `pyproject.toml`.
- `download_dataset.sh` downloads supported datasets.
- `download_models.sh` downloads configured torchvision checkpoints.
- `prepare_expeca_author_layout.sh` creates the dataset paths expected by the
  author's ExPECA Dockerfile.
- `setup_expeca_notebook_env.sh` installs notebook-only ExPECA dependencies
  into the uv environment.
- `check_expeca_cpu_prereqs.sh` checks local prerequisites for the ExPECA
  public-IP CPU baseline.
- `build_expeca_cpu_images.sh` builds the author's CPU edge-server and
  edge-device images with configurable registry tags.
- `push_expeca_cpu_images.sh` pushes those CPU images to your registry.
- `build_expeca_gpu_server_image.sh` builds a CUDA-enabled edge-server image.
- `push_expeca_gpu_server_image.sh` pushes the CUDA-enabled edge-server image.

They read stable defaults from `config/defaults.env` and active run choices from
`config/experiment.env`. Values exported in the shell still override both files.

```bash
SML_ARCH=resnet34 scripts/download_models.sh
```

## Order

1. Sync the Python environment:

   ```bash
   scripts/setup_env.sh
   source .venv/bin/activate
   ```

2. Download datasets:

   ```bash
   scripts/download_dataset.sh --all
   ```

   Useful focused variants:

   ```bash
   scripts/download_dataset.sh --imagenette
   scripts/download_dataset.sh --imagenetv2
   ```

3. Download model checkpoints:

   ```bash
   scripts/download_models.sh
   ```

   To download every model currently supported by the repo:

   ```bash
   scripts/download_models.sh --all
   ```

The default model config saves torchvision checkpoints for:

- SML: `mobilenet_v3_large`
- LML: `wide_resnet50_2`

## Thesis Dataset

The thesis reproduction requires the real ImageNetV2 Matched Frequency validation
set, not the placeholder directory created for Docker compatibility.

Download and validate it with:

```bash
scripts/download_dataset.sh --imagenetv2
```

Expected validated layout:

```text
data/datasets/imagenetV2/matched-frequency-format-val/
  0/
  1/
  ...
  999/
```

The script checks for `1000` class folders and `10000` images. Use this path in
`config/experiment.env`:

```env
SAMPLE_PATH=data/datasets/imagenetV2/matched-frequency-format-val
```

## ExPECA Public-IP CPU Baseline

Before using the ExPECA public-IP notebook, prepare the local assets and log in
to your registry:

```bash
scripts/download_dataset.sh
scripts/download_models.sh --all
scripts/prepare_expeca_author_layout.sh
scripts/setup_expeca_notebook_env.sh

docker login
```

Then set your registry namespace in `config/experiment.env`:

```env
EXPECA_IMAGE_NAMESPACE=YOUR_DOCKERHUB_USERNAME_OR_REGISTRY_NAMESPACE
```

The default CPU image tags and platform live in `config/defaults.env`. Change
those only when publishing a new tag or targeting a different platform.

Build and push:

```bash
scripts/check_expeca_cpu_prereqs.sh
scripts/build_expeca_cpu_images.sh
scripts/push_expeca_cpu_images.sh
```

Then follow `docs/expeca_public_ip_cpu_baseline.md` to edit the notebook image
names, start ExPECA containers, set `config/experiment.env`, and run
`src/run_expeca_public_ip_test.py`.

Set `CONTROLLER_MAX_SAMPLES=all` in `config/experiment.env` for a full
dataset run. Use a small integer such as `4` for a quick remote smoke test.

After one public-IP run works, sweep server-side batch sizes with:

```bash
python src/run_expeca_batch_grid.py
```

The grid is controlled by `BATCH_SIZE_GRID`, `CONTROLLER_BATCH_SIZE`,
`CONTROLLER_BATCH_SIZE_GRID`, and `BATCH_GRID_PAIR_MODE` in
`config/experiment.env`.

Combine CPU/GPU grid summaries with:

```bash
python src/compare_grid_results.py
```

That command writes long-format comparison CSVs, m-by-n pivot CSVs, and optional
plots under `results/comparison_expeca_public_ip/`.

## ExPECA GPU Server Prep

The edge-device image can stay CPU. For GPU experiments, build and push only the
CUDA-enabled edge-server image:

```bash
# In config/experiment.env:
# EXPECA_GPU_EDGE_SERVER_IMAGE_TAG=gpu-amd64-001

scripts/build_expeca_gpu_server_image.sh
scripts/push_expeca_gpu_server_image.sh
```

Then in `notebooks/ExPECA_HI_setup_Public_IP.ipynb`, set:

```python
EDGE_SERVER_IMAGE_TAG = "gpu-amd64-001"
EDGE_SERVER_DEVICE = "cuda"
```

Keep the edge-device image/device on CPU unless you intentionally want to test a
GPU edge-device too.
