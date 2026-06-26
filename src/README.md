# Python Utilities

This folder contains Python utilities for local development and experiment
orchestration. Shell setup/download helpers live in `scripts/`.

## Local Smoke Test

Run the local Python-script path with one small fixed batch:

```bash
source .venv/bin/activate
python src/run_local_smoke_test.py
```

Set `DEVICE` in `config/experiment.env` to choose compute mode:

```text
DEVICE=auto  # use CUDA if available, otherwise CPU
DEVICE=cpu   # force CPU
DEVICE=cuda  # require CUDA, fail if unavailable
```

The smoke test starts:

```text
edge_server.py
edge_device.py
```

Then it sends one configured sample batch through:

```text
controller utility -> edge_device.py -> edge_server.py
```

It reads defaults from `config/experiment.env`. The edge-device service writes
raw results to `results/EdgeDevice_results.csv`.

For this local utility:

- `CONTROLLER_BATCH_SIZE` is how many images the utility sends to the edge device.
- `BATCH_SIZE` is how many offloaded images the edge device sends to the edge server.
- `FLUSH_FINAL_BATCH=true` sends any final partial edge-server batch at the end
  of the request.

## Results Analysis

Summarize a result CSV into derived timings and a readable markdown report:

```bash
python src/analyze_results.py
```

Outputs:

```text
results/analysis/timing_results.csv
results/analysis/summary.md
```

The analyzer expects `results/EdgeDevice_results.csv` and writes into
`results/analysis/`.
