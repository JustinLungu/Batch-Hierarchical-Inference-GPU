# Shell Scripts

This folder contains shell helpers for preparing local experiments:

- `setup_env.sh` syncs the uv environment from `pyproject.toml`.
- `download_dataset.sh` downloads the configured local dataset.
- `download_models.sh` downloads configured torchvision checkpoints.
- `prepare_expeca_author_layout.sh` creates the dataset paths expected by the
  author's ExPECA Dockerfile.
- `check_expeca_cpu_prereqs.sh` checks local prerequisites for the ExPECA
  public-IP CPU baseline.
- `build_expeca_cpu_images.sh` builds the author's CPU edge-server and
  edge-device images with configurable registry tags.
- `push_expeca_cpu_images.sh` pushes those CPU images to your registry.

They read defaults from `config/experiment.env`. You can override that file per
command:

```bash
CONFIG_FILE=config/experiment.env scripts/download_dataset.sh
SML_ARCH=resnet34 scripts/download_models.sh
```

## Order

1. Sync the Python environment:

   ```bash
   scripts/setup_env.sh
   source .venv/bin/activate
   ```

2. Download a small local dataset:

   ```bash
   scripts/download_dataset.sh
   ```

3. Download model checkpoints:

   ```bash
   scripts/download_models.sh
   ```

   To download every model currently supported by the repo:

   ```bash
   scripts/download_models.sh --all
   ```

The default config downloads Imagenette 160px and saves torchvision checkpoints
for:

- SML: `mobilenet_v3_large`
- LML: `wide_resnet50_2`

## ExPECA Public-IP CPU Baseline

Before using the author's ExPECA public-IP notebook, prepare the original Docker
layout and build/push images:

```bash
scripts/download_dataset.sh
scripts/download_models.sh --all
scripts/prepare_expeca_author_layout.sh

EXPECA_IMAGE_NAMESPACE=your_dockerhub_user scripts/check_expeca_cpu_prereqs.sh
EXPECA_IMAGE_NAMESPACE=your_dockerhub_user scripts/build_expeca_cpu_images.sh
EXPECA_IMAGE_NAMESPACE=your_dockerhub_user scripts/push_expeca_cpu_images.sh
```

Then follow `docs/expeca_public_ip_cpu_baseline.md`.
