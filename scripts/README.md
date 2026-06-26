# Local Setup Scripts

These scripts prepare the local Python-script workflow:

```text
controller -> edge_device.py -> edge_server.py
```

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
