# Simple I-JEPA

A simple and efficient PyTorch implementation of **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)**, plus a **Gated I-JEPA** variant with differentiable hard-concrete token gating.

The code now uses **Hydra** for configuration and a single, unified config tree with clearly separated sub-sections (model, data, optimization, EMA, etc.). Every knob can be overridden from the command line via `key=value` overrides.

---

## Results

The baseline model was pre-trained on 100,000 unlabeled images from the `STL-10` dataset. For evaluation, logistic regression was trained on frozen features obtained from 5k train images and evaluated on 8k test images (also from `STL-10`). 

Linear probing is done with scikit-learn’s `LogisticRegression` (wrapped in a `CalibratedClassifierCV`), using features from the encoder at image resolution `96x96`.

More detailed evaluation steps and results for [STL10](https://github.com/filipbasara0/simple-ijepa/blob/main/notebooks/linear-probing-stl.ipynb) can be found in the `notebooks/` directory. 

| Dataset | Approach | Encoder           | Emb. dim | Patch size | Num. targets | Batch size | Epochs | Top 1% |
|---------|----------|-------------------|----------|------------|--------------|------------|--------|--------|
| STL10   | I-JEPA   | VisionTransformer | 512      | 8          | 4            | 256        | 100    | 77.07  |

All experiments were done using a very small and shallow VisionTransformer (≈11M parameters) with the following parameters:

* embedding dimension: `512`
* depth (number of transformer layers): `6`
* number of heads: `6`
* MLP dim: `2 * embedding dimension`
* patch size: `8`
* number of targets: `4`

The mask generator in the baseline I-JEPA variant is inspired by the original paper, but slightly simplified.

---

## Installation

To setup the code, clone the repository, optionally create a venv and install the package:

1. `git clone git@github.com:filipbasara0/simple-ijepa.git`
2. `cd simple-ijepa`
3. Create virtual environment: `virtualenv -p python3.10 env`
4. Activate virtual environment: `source env/bin/activate`
5. Install the package: `pip install .`

> Note: Distributed training uses `torch.distributed` with the `nccl` backend. You’ll generally want a CUDA build of PyTorch and at least one GPU.

---

## Training

Training is driven by a single Hydra-based entrypoint:

* Python module: `scripts.train_ddp`
* Shim script: `run_ddp.py` (just calls `scripts.train_ddp.main`)

Because training is DDP-based, you should launch it via `torchrun` (or equivalent).

### Baseline I-JEPA (with STL-10 masks)

Single-GPU example (DDP with 1 process):

`torchrun --standalone --nproc_per_node=1 run_ddp.py variant=baseline`

Multi-GPU example (4 GPUs on a single node):

`torchrun --standalone --nproc_per_node=4 run_ddp.py variant=baseline`

Roughly matching the original STL-10 experiment (adjust batch size / epochs as needed):

`torchrun --standalone --nproc_per_node=4 run_ddp.py variant=baseline dataloader.batch_size=256 optim.num_epochs=100`

### Gated I-JEPA

To use the Gated I-JEPA variant, set `variant=gated` and tweak the gating knobs under `model.*`:

Example:

`torchrun --standalone --nproc_per_node=4 run_ddp.py variant=gated model.lambda_gates=1.0 model.gate_exp_alpha=4.0`

If you want to shift gating *inside* the VisionTransformer (after a specific block):

`torchrun --standalone --nproc_per_node=4 run_ddp.py variant=gated model.gate_layer_index=2 model.gate_location=post`

(Details on these knobs are in the “Configuration / knobs” section below.)

### Resume from checkpoint

To resume from a training checkpoint, use:

`torchrun --standalone --nproc_per_node=4 run_ddp.py logging.ckpt_path=./models/training_model_ddp.pth`

(or the gated version `training_model_gated_ddp.pth` if `variant=gated`).

### STL-10 linear probing during training

The trainer periodically runs STL-10 linear probing via `STL10Eval` (logistic regression on frozen features):

* It runs automatically on **rank 0** whenever `global_step % logging.log_every_n_steps == 0`.
* For the baseline variant, it evaluates both the **teacher encoder** and (for comparison) the **student encoder**.
* For the gated variant, it evaluates:
  * Student encoder + learned gates (`mode="gated"`)
  * Student encoder with all gates open (`mode="all_open"`)

You can control evaluation frequency via `logging.log_every_n_steps`.

---

## Gated model analysis utilities

For a trained Gated I-JEPA model, you can run analysis experiments (average gate maps, noise vs real images, random-gate ablations) via:

`python -m simple_ijepa.gate_experiments --ckpt_path ./models/training_model_gated_ddp.pth --dataset_path ./data --split train --num_batches 50 --out_dir ./gate_analysis`

This script will:

* Rebuild a `GatedIJEPA` model and load the checkpoint.
* Run multiple experiments:
  * Average gate map over a dataset.
  * Noise vs real images (open fractions).
  * Random-gate ablation vs learned gates vs all-open baseline.
* Save tensors and plots into `out_dir`:
  * `avg_gate_map.pt`, `avg_gate_map.png`
  * `open_fracs.npy`, `open_fracs_hist.png`
  * `noise_vs_real_stats.pt`, `noise_vs_real_open_frac.png`
  * `random_gate_ablation_stats.pt`, `random_gate_ablation_losses.png`

---

## Configuration / knobs (Hydra)

The entire training configuration is a single Hydra config tree rooted at `TrainConfig`. You can override **any** field from the command line using `section.key=value` syntax.

### Top-level

Section: root

| Key      | Default     | Description                                   |
|----------|-------------|-----------------------------------------------|
| variant  | `"baseline"`| `"baseline"` or `"gated"` model variant.      |

### Model section (`model.*`) – `IJEPAConfig`

Architecture + gating configuration shared by both baseline and gated variants.

| Key               | Default | Description                                                                  |
|-------------------|---------|------------------------------------------------------------------------------|
| model.image_size  | `96`    | Input image size (assumed square, `H=W=image_size`).                        |
| model.patch_size  | `8`     | Patch size (image is split into `(image_size / patch_size)^2` patches).     |
| model.hidden_dim  | `512`   | Transformer embedding dimension.                                            |
| model.depth       | `6`     | Number of transformer layers in the encoder.                                |
| model.heads       | `6`     | Number of attention heads in the encoder.                                   |
| model.predictor_depth  | `6` | Depth of the predictor VisionTransformer.                                   |
| model.predictor_heads  | `6` | Number of heads in the predictor.                                           |
| model.num_targets | `4`     | Number of target blocks for baseline I-JEPA masking.                        |

Gating-specific knobs (used only when `variant="gated"`):

| Key                    | Default | Description                                                                                 |
|------------------------|---------|---------------------------------------------------------------------------------------------|
| model.lambda_gates     | `1.0`   | Weight of the gate sparsity penalty.                                                        |
| model.gate_exp_alpha   | `4.0`   | Exponential factor in the gate penalty: larger values penalize open gates more strongly.   |
| model.gate_layer_index | `null`  | If `null`: gate on final encoder tokens (outside encoder). If `k`: gate after block `k`.   |
| model.gate_location    | `"post"`| Where inside the chosen block to apply gating: `"attn"`, `"skip"`, or `"post"`.            |
| model.gate_beta        | `2/3`   | Hard-concrete temperature (sharper gates when smaller).                                     |
| model.gate_gamma       | `-0.1`  | Unused in this simplified gate, kept for API compatibility.                                |
| model.gate_zeta        | `1.1`   | Unused in this simplified gate, kept for API compatibility.                                |

### Data section (`data.*`) – `DataConfig`

| Key                 | Default    | Description                            |
|---------------------|------------|----------------------------------------|
| data.dataset_path   | `"./data"` | Root directory for datasets.          |
| data.dataset_name   | `"stl10"`  | Dataset name (currently only `stl10`).|

### Optimization section (`optim.*`) – `OptimConfig`

| Key                      | Default   | Description                                             |
|--------------------------|-----------|---------------------------------------------------------|
| optim.num_epochs         | `100`     | Number of training epochs.                             |
| optim.learning_rate      | `3e-4`    | Learning rate.                                         |
| optim.weight_decay       | `1e-5`    | Weight decay for optimizer.                            |
| optim.fp16_precision     | `true`    | Use mixed-precision (`torch.cuda.amp`) if `true`.      |

### Dataloader section (`dataloader.*`) – `DataloaderConfig`

| Key                      | Default | Description                                  |
|--------------------------|---------|----------------------------------------------|
| dataloader.batch_size    | `1024`  | Per-GPU batch size.                          |
| dataloader.num_workers   | `8`     | Number of dataloader workers per process.    |

### EMA section (`ema.*`) – `EMAConfig`

| Key                             | Default | Description                                                                                       |
|---------------------------------|---------|---------------------------------------------------------------------------------------------------|
| ema.gamma                       | `0.996` | Initial EMA coefficient for teacher parameters.                                                  |
| ema.update_gamma_after_step    | `1`     | Do hard copy (`teacher ← student`) up to this step, then start EMA updates.                      |
| ema.update_gamma_every_n_steps | `1`     | Frequency (in steps) of EMA updates after `update_gamma_after_step`.                             |

`gamma` is also decayed over training via `train.update_gamma(...)` to gradually increase teacher lag.

### Logging / checkpoint section (`logging.*`) – `LoggingConfig`

| Key                          | Default        | Description                                                                 |
|------------------------------|----------------|-----------------------------------------------------------------------------|
| logging.save_model_dir       | `"./models"`   | Directory where training and encoder checkpoints are saved.                |
| logging.log_every_n_steps    | `98`           | Step interval for logging, checkpointing, and STL-10 evaluation.           |
| logging.ckpt_path            | `null`         | Optional path to `.pth` checkpoint to resume training.                     |

The trainer saves:

* Full training checkpoint (student + EMA teacher + predictor) as:
  * Baseline: `training_model_ddp.pth`
  * Gated: `training_model_gated_ddp.pth`
* EMA teacher encoder only:
  * Baseline: `encoder_ddp.pth`
  * Gated: `encoder_gated_ddp.pth`

### Debug section (`debug.*`) – `DebugConfig`

| Key                    | Default | Description                                                                                             |
|------------------------|---------|---------------------------------------------------------------------------------------------------------|
| debug.save_debug_masks | `false` | If `true` and `variant="gated"`, rank 0 saves patch-masked image visualizations and SSIM heatmaps.     |

When enabled, the trainer:

* Saves original vs masked images using the current gate values into `save_model_dir/debug_masks/`.
* Computes an SSIM-style similarity matrix between the gate-MLP input tokens and saves:
  * `debug_ssim/*_token_ssim.pt`
  * `debug_ssim/*_token_ssim.png`

---

## Examples of CLI overrides

A few concrete examples to illustrate how to use the knobs:

1. Gated model with stronger gate penalty and internal gating at block 3:

   `torchrun --standalone --nproc_per_node=4 run_ddp.py variant=gated model.lambda_gates=2.0 model.gate_exp_alpha=6.0 model.gate_layer_index=3 model.gate_location=attn`

2. Smaller batch size, more epochs, different learning rate:

   `torchrun --standalone --nproc_per_node=2 run_ddp.py variant=baseline dataloader.batch_size=256 optim.num_epochs=300 optim.learning_rate=1e-4`

3. Change dataset path and enable debug mask visualizations:

   `torchrun --standalone --nproc_per_node=1 run_ddp.py variant=gated data.dataset_path=/mnt/datasets/stl10 debug.save_debug_masks=true`

4. Train with full FP32 (no mixed precision):

   `torchrun --standalone --nproc_per_node=1 run_ddp.py variant=baseline optim.fp16_precision=false`

---

## Citation

If you use this codebase in your research, please consider citing the original I-JEPA paper:

```
@misc{assran2023selfsupervisedlearningimagesjointembedding,
      title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture}, 
      author={Mahmoud Assran and Quentin Duval and Ishan Misra and Piotr Bojanowski and Pascal Vincent and Michael Rabbat and Yann LeCun and Nicolas Ballas},
      year={2023},
      eprint={2301.08243},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2301.08243}, 
}
```
