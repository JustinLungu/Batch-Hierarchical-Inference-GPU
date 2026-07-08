# Thesis Summary: Batch Hierarchical Inference for Edge Computing

This document summarizes the important technical content from Henrik Leevi Matias
Luukkanen's thesis, *Batch Hierarchical Inference for Edge Computing:
Implementation of Batch Hierarchical Inference on the ExPECA Testbed*.

The purpose of this note is to preserve the thesis context needed for this
repository without repeatedly reopening the full PDF.

## Research Question

The thesis studied how batch processing affects hierarchical inference in a real
edge-computing testbed.

The central question was:

```text
How does batch processing influence hierarchical inference in terms of accuracy,
latency, throughput, and communication efficiency in real-world edge computing
environments?
```

The motivation was that conventional hierarchical inference can offload uncertain
samples immediately, one by one. This can preserve accuracy but may create high
communication overhead and poor server utilization. The thesis investigated whether
collecting samples into batches before offloading could improve efficiency without
damaging accuracy too much or introducing unacceptable latency.

## System Architecture

The implemented system used a two-tier hierarchical inference architecture:

```text
Edge Device (ED)
  runs a small model first
  decides whether the sample is simple or complex
  keeps simple samples locally
  offloads complex samples

Edge Server (ES)
  receives offloaded samples
  runs a large model
  returns predictions/results to the edge device
```

The thesis framework had three main software components:

- Edge Device service
- Edge Server service
- Controller

The Edge Device and Edge Server were implemented as FastAPI services and packaged
in Docker containers. The controller orchestrated experiments by configuring the
services, dispatching input samples, and collecting logs/results.

In the ExPECA deployment, the controller was embedded inside the edge-device
container.

## Dataset

The thesis used ImageNetV2 Matched Frequency validation set.

Important details:

- 10,000 image samples
- 1,000 ImageNet classes
- Chosen because it preserves the original ImageNet class distribution while adding
  mild distribution shift
- Used as an unseen inference workload for pretrained ImageNet models

The thesis focused only on image classification.

## Models

The system used one small model on the Edge Device and one large model on the Edge
Server.

### Small Model

The Edge Device model was:

```text
MobileNetV3-Large-V2
```

The thesis describes it as:

- Low memory footprint
- Fast inference
- Suitable for Raspberry Pi 4 constraints
- Around 5.5M parameters
- Around 0.22 GFLOPS
- Around 21.1 MB

Its role was to provide fast local inference and confidence scores for offloading
decisions.

### Large Model

The Edge Server model was:

```text
ViT-H-14-SWAG-E2E-V1
```

The thesis describes it as:

- High-capacity vision transformer
- Around 633.5M parameters
- Around 1016.72 GFLOPS
- Around 2416.6 MB
- Used as the high-accuracy model for offloaded samples

This large model was computationally heavy and ran on CPU in the thesis
experiments.

## Hardware

### Edge Device

The Edge Device was a Raspberry Pi 4 Model B.

The thesis describes it as:

- Broadcom BCM2711, Cortex-A72 class CPU
- 8 GB LPDDR4 RAM
- No dedicated GPU, NPU, or TPU

The Raspberry Pi represented a constrained IoT/edge node.

### Edge Server

The Edge Server was hosted on a Dell PowerEdge R450 in the ExPECA testbed.

The thesis describes it as:

- Intel Xeon Silver 4310 CPU
- 12-core CPU at 2.10 GHz
- 32 GB DDR4 RAM
- No dedicated GPU

All server-side large-model inference was CPU-based.

## Network

The experiments used the ExPECA testbed with a dedicated Ericsson private 5G link.

The Edge Device communicated through an Advantech ICR-4453 5G router. The Edge
Server was located in the onsite network/core side. The private 5G setup was chosen
to provide realistic but controlled network behavior.

The thesis notes that this environment avoids many sources of public cellular
network variability, such as congestion, handovers, policy throttling, and multi-hop
public network delays.

## Timing and Logging

The thesis emphasized fine-grained timestamp logging.

The ExPECA testbed provided PTP-based clock synchronization across nodes, allowing
timestamps on the Edge Device and Edge Server to be compared.

The implementation logged per-sample events using UUIDs. Logged events included:

- SML inference start/end
- Offloading decision time
- Batch queue / buffering time
- Network send and receive timestamps
- LML inference start/end
- Result send/receive timestamps

After each experiment, logs were collected and post-processed to compute derived
metrics such as total per-sample latency, queueing/buffering delay, batch size
distributions, number of offload transmissions, and throughput.

## Offloading Strategies

The thesis implemented four offloading decision strategies.

### Never Offload

All samples were processed only by the small model on the Edge Device.

Purpose:

- Minimum-latency baseline
- No communication overhead
- Lowest expected accuracy

### Always Offload

All samples were processed locally by the small model and then also offloaded to the
large model on the Edge Server.

Purpose:

- Accuracy upper bound
- Highest expected latency and communication cost

### Fixed Threshold

Samples were offloaded if the small model's confidence was below a fixed threshold.

The threshold was chosen using a grid search over confidence values. The thesis used
a cost function that penalized incorrect local predictions and unnecessary offloading.
The selected fixed threshold was approximately:

```text
0.3888
```

### Adaptive Threshold

The adaptive method adjusted the offloading threshold over time based on feedback
about whether previous local predictions were correct.

The method discretized confidence values and updated weights using an online
learning-style rule. It started from a conservative threshold around:

```text
0.75
```

Over time, it converged near the fixed-threshold optimum, around:

```text
0.39
```

The thesis used adaptive thresholding as the basis for the batching configurations.

## Batching Strategies

The thesis considered three offload transmission styles.

### Send Individually

Each sample selected for offloading was sent immediately to the Edge Server.

This minimized waiting time for individual samples but increased the number of
network transmissions.

### Dynamic Batching

The Edge Device processed a fixed-size group of input samples. After local inference,
the samples that needed offloading were sent together as one offload batch.

This decoupled the local input batch size from the actual number of samples
offloaded, because not every sample in the input batch necessarily required
offloading.

### Size-Based Batching

The Edge Device accumulated offloaded samples until a target offload batch size was
reached, then transmitted the batch to the Edge Server.

This reduced communication overhead but could introduce waiting time while the
buffer filled.

## Experimental Configurations

The thesis evaluated seven main configurations.

| Config | Offloading Logic | Batching Logic | Purpose |
|---|---|---|---|
| 001 | Never Offload | None | Minimum-latency local baseline |
| 002 | Always Offload | Send individually | Accuracy upper bound |
| 003 | Fixed Threshold | Send individually | Static selective offloading |
| 004 | Adaptive Threshold | Send individually | Adaptive selective offloading |
| 005 | Adaptive Threshold | Dynamic batching, initial batch 5 | Small batch |
| 006 | Adaptive Threshold | Dynamic batching, initial batch 15 | Moderate batch |
| 007 | Adaptive Threshold | Dynamic batching, initial batch 45 | Large batch |

The dataset, models, software environment, and hardware were kept constant across
these configurations so the effects of offloading and batching strategy could be
compared.

## Evaluation Metrics

The thesis evaluated:

- Accuracy
- Offloading behavior / offloading ratio
- Per-sample latency
- Throughput
- Communication overhead
- Batch size distributions
- Number of offloading transmissions

Latency for locally processed samples included only local small-model inference.

Latency for offloaded samples included:

- SML inference
- Offloading decision time
- Network transfer
- Buffering/queueing delay
- LML inference
- Result return

Average, median, and 95th percentile latency were considered.

## Main Results

### Accuracy

The never-offload configuration had the lowest accuracy because it relied only on the
small Edge Device model.

The always-offload configuration had the highest accuracy because every sample used
the large Edge Server model.

Fixed-threshold and adaptive-threshold configurations improved accuracy over local
only inference while avoiding offloading every sample.

The thesis found that batching did not significantly reduce classification accuracy.
This supported the claim that batching can improve system efficiency without
damaging inference quality, as long as the offloading decisions remain effective.

### Offloading Behavior

The fixed threshold offloaded about 39% of samples.

The adaptive threshold converged toward a similar offloading ratio over time.

For batched adaptive configurations, threshold updates happened at batch level rather
than per-sample level. The thesis concluded that this coarser update frequency did not
materially damage offloading behavior.

### Latency

The thesis reported:

```text
Never offload average latency: about 0.25s
Always offload average latency: about 4.16s
```

The always-offload case had the highest latency because every sample paid the cost of:

- local SML inference
- network transfer
- remote LML inference
- result return

For batched configurations, larger batch sizes increased latency due to buffering and
queueing.

The thesis specifically identified two major contributors in the latency breakdown:

1. Edge Device offloading buffer delay
2. Edge Server processing, including queueing delays as batch size grew

Thus, batching improved efficiency but introduced extra per-sample waiting time.

### Throughput

Throughput increased as batch size increased.

The thesis concluded that batching improved resource utilization by reducing
per-sample communication and allowing more efficient server-side processing.

Among the evaluated batched configurations, the moderate batch configuration was
considered the best balance:

```text
Config 006: adaptive threshold + dynamic batching with initial batch size 15
```

This configuration improved throughput substantially compared with non-batched
selective offloading while avoiding the larger queuing delays seen with batch size 45.

### Communication Efficiency

The thesis measured communication efficiency by counting the number of offload
transmissions required to process a standard workload.

The relevant comparison was:

| Config | Initial Batch | Average Offload Batch | Offload Transmissions | Efficiency Gain |
|---|---:|---:|---:|---:|
| 004 | 1 sample | 0.39 samples | 3900 | baseline |
| 005 | 5 samples | 1.99 samples | 2000 | about 49% fewer transmissions |
| 006 | 15 samples | 6.03 samples | 667 | about 83% fewer transmissions |
| 007 | 45 samples | 18.00 samples | 225 | about 94% fewer transmissions |

All adaptive configurations offloaded a similar total number of samples, but batching
reduced how many separate transmissions were required.

The thesis emphasized that Config 006 reduced offload transmissions by over 80%
compared with individual offloading while preserving high accuracy and avoiding the
worst latency penalties.

## Thesis Conclusions

The thesis concluded that batch processing can improve hierarchical inference
efficiency without significantly compromising accuracy.

The central conclusions were:

- Batching reduces communication overhead.
- Batching improves throughput.
- Batching does not significantly reduce accuracy when paired with effective
  offloading decisions.
- Larger batches introduce extra latency due to buffering and queueing.
- A moderate batch size gives the best practical balance.
- In the thesis experiments, batch size 15 was the best compromise.
- Very large batches, such as batch size 45, gave stronger communication reduction
  but introduced excessive delay.

The final recommendation was not to maximize batch size blindly, but to choose a
moderate batch size based on the desired latency-throughput tradeoff.

## Limitations

The thesis identified several limitations:

- The Edge Device had no GPU/NPU/TPU.
- The Edge Server also had no GPU.
- All large-model inference was CPU-based.
- The task was limited to image classification.
- The experiments used ImageNetV2 only.
- The ExPECA private 5G environment was controlled and may not reflect public
  cellular network variability.
- Detailed logging introduced overhead that may inflate latency compared with a
  production-optimized system.

## Future Work Proposed

The thesis proposed several directions for future research:

- Evaluate hardware acceleration using GPU, TPU, or NPU.
- Introduce context-aware batching based on runtime conditions.
- Adapt batching based on input arrival rate, network congestion, or power state.
- Extend the architecture to multiple inference tiers, such as edge-to-cloud.
- Test other data modalities beyond image classification.

The most relevant future-work point for this repository is GPU acceleration on the
Edge Server, because the thesis server-side LML inference was CPU-only despite using a
very large model.

