# Shell Scripts

This folder contains setup helpers for thesis reproduction on ExPECA.

They read stable defaults from `config/defaults.env` and runtime settings from
`config/experiment.env`. Values exported in the shell still override both files.

## Setup Order

1. Sync the Python environment:

   ```bash
   scripts/setup_env.sh
   source .venv/bin/activate
   ```

2. Download the thesis dataset:

   ```bash
   scripts/download_dataset.sh --imagenetv2
   ```

3. Download model checkpoints:

   ```bash
   scripts/download_models.sh --all
   ```

   The default configured pair is the thesis pair:

   ```text
   SML: mobilenet_v3_large
   LML: vit_h_14
   ```

4. Prepare the dataset layout expected by the original ExPECA Dockerfiles:

   ```bash
   scripts/prepare_expeca_author_layout.sh
   ```

5. Install ExPECA notebook dependencies:

   ```bash
   scripts/setup_expeca_notebook_env.sh
   ```

## CPU Images

Set your registry namespace in `config/experiment.env`:

```env
EXPECA_IMAGE_NAMESPACE=YOUR_DOCKERHUB_USERNAME_OR_REGISTRY_NAMESPACE
```

Build and push CPU images:

```bash
scripts/check_expeca_cpu_prereqs.sh
scripts/build_expeca_cpu_images.sh
scripts/push_expeca_cpu_images.sh
```

## GPU Edge-Server Image

For GPU reproduction, keep the edge-device CPU image and build only the CUDA
edge-server image:

```bash
scripts/build_expeca_gpu_server_image.sh
scripts/push_expeca_gpu_server_image.sh
```

## Main Runbook

Follow:

```text
docs/thesis_reproduction.md
```

The main command is:

```bash
python src/run_thesis_reproduction.py --dry-run
python src/run_thesis_reproduction.py
```
