# simple_ijepa/gate_experiments.py
#
# Utilities to probe a trained GatedIJEPA model and understand whether
# its learned gates are content-aware or closer to random/positional.
#
# Experiments:
#   1) Average gate map over a dataset (is there strong positional bias?)
#   2) Noise vs real images (do gates behave differently on noise?)
#   3) Random-gate ablation (functional test: learned vs random gates
#      at the same sparsity, plus an all-open baseline).
#
# Outputs:
#   - Numeric stats saved as .pt / .npy
#   - Visualizations saved as .png in the output directory:
#       * avg_gate_map.png
#       * open_fracs_hist.png
#       * noise_vs_real_open_frac.png
#       * random_gate_ablation_losses.png
#
# Usage example:
#   python -m simple_ijepa.gate_experiments \
#       --ckpt_path ./models_gated_ddp/training_model_gated_ddp.pth \
#       --dataset_path ./data \
#       --split train \
#       --num_batches 50 \
#       --out_dir ./gate_analysis

import argparse
import os
from typing import Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import STL10
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from simple_ijepa.transformer import VisionTransformer
from simple_ijepa.ijepa_gated import GatedIJEPA
from simple_ijepa.utils import training_transforms


def build_gated_model_from_ckpt(
    ckpt_path: str,
    device: torch.device,
    img_size: int = 96,
    patch_size: int = 8,
    dim: int = 512,
    depth: int = 6,
    heads: int = 6,
    mlp_mult: int = 2,
    lambda_gates: float = 1e-3,
    gate_exp_alpha: float = 5.0,
) -> GatedIJEPA:
    """
    Rebuild a GatedIJEPA model with the same architecture as in training
    and load a checkpoint.

    We assume the same hyperparameters as your training script:
        image_size=96, patch_size=8, dim=512, depth=6, heads=6, mlp_dim=2*dim
    """
    mlp_dim = dim * mlp_mult

    encoder = VisionTransformer(
        image_size=img_size,
        patch_size=patch_size,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
    )

    model = GatedIJEPA(
        encoder=encoder,
        hidden_emb_dim=dim,
        img_size=img_size,
        patch_size=patch_size,
        predictor_depth=depth,
        predictor_heads=heads,
        lambda_gates=lambda_gates,
        gate_exp_alpha=gate_exp_alpha,
    )

    print(f"Loading checkpoint from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def compute_gate_maps(
    model: GatedIJEPA,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-sample gate maps (expected open probabilities) for a number
    of batches, and return:

      - avg_map:      (H_p, W_p) average gate_probs over all samples
      - open_fracs:  (num_samples,) expected fraction of open gates per sample
    """
    all_maps = []
    all_open_fracs = []

    H_p = model.num_patches_per_dim
    W_p = model.num_patches_per_dim

    for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Computing gate maps")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        # Manually run the gating path to get full gate_probs for the batch
        student_tokens = model.context_encoder(images)              # (B, N, D)
        log_alpha = model.gate_mlp(student_tokens).squeeze(-1)      # (B, N)
        gate_values, gate_probs = model.gate(log_alpha, training=False)  # (B, N)

        B, N = gate_probs.shape
        assert H_p * W_p == N, f"Mismatch: H_p * W_p = {H_p*W_p}, N = {N}"

        # Reshape each sample's probs to (H_p, W_p)
        gate_maps = gate_probs.view(B, H_p, W_p).cpu().numpy()      # (B, H_p, W_p)
        all_maps.append(gate_maps)

        # Per-sample open fraction f_b = mean_i pi_{b,i}
        open_fracs = gate_probs.mean(dim=1).cpu().numpy()           # (B,)
        all_open_fracs.append(open_fracs)

    if not all_maps:
        raise RuntimeError("No batches processed in compute_gate_maps.")

    all_maps = np.concatenate(all_maps, axis=0)          # (num_samples, H_p, W_p)
    all_open_fracs = np.concatenate(all_open_fracs, axis=0)  # (num_samples,)

    avg_map = all_maps.mean(axis=0)                      # (H_p, W_p)
    return avg_map, all_open_fracs


@torch.no_grad()
def noise_vs_real_experiment(
    model: GatedIJEPA,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 10,
) -> Dict[str, float]:
    """
    Compare gating on real STL-10 images vs pure noise images
    of the same shape.

    Returns a dict with means/stds of open fraction for real vs noise.
    """
    real_open_fracs = []
    noise_open_fracs = []

    for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Noise vs real")):
        if batch_idx >= max_batches:
            break

        images = images.to(device)
        B, C, H, W = images.shape

        # --- Real images ---
        student_tokens_real = model.context_encoder(images)
        log_alpha_real = model.gate_mlp(student_tokens_real).squeeze(-1)
        _, gate_probs_real = model.gate(log_alpha_real, training=False)
        real_open_fracs.append(gate_probs_real.mean(dim=1).cpu().numpy())  # (B,)

        # --- Pure noise images (same shape) ---
        noise = torch.randn_like(images)  # Gaussian noise
        student_tokens_noise = model.context_encoder(noise)
        log_alpha_noise = model.gate_mlp(student_tokens_noise).squeeze(-1)
        _, gate_probs_noise = model.gate(log_alpha_noise, training=False)
        noise_open_fracs.append(gate_probs_noise.mean(dim=1).cpu().numpy())

    real_open_fracs = np.concatenate(real_open_fracs, axis=0)
    noise_open_fracs = np.concatenate(noise_open_fracs, axis=0)

    stats = {
        "real_mean": float(real_open_fracs.mean()),
        "real_std": float(real_open_fracs.std()),
        "noise_mean": float(noise_open_fracs.mean()),
        "noise_std": float(noise_open_fracs.std()),
        "num_samples": int(real_open_fracs.shape[0]),
        "real_vals": real_open_fracs,
        "noise_vals": noise_open_fracs,
    }
    return stats


@torch.no_grad()
def compute_pred_loss_with_given_gates(
    model: GatedIJEPA,
    images: torch.Tensor,
    gate_values: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the prediction loss L_pred on a batch for a *given* gate map r.

    This reimplements the prediction loss from GatedIJEPA.forward, but with
    externally supplied gate_values:

        r_{b,i} \in [0,1], shape (B, N).

    Args:
        model:        Trained GatedIJEPA.
        images:       (B, C, H, W) on the right device.
        gate_values:  (B, N) tensor of gates in [0,1].

    Returns:
        pred_loss: scalar tensor.
    """
    device = images.device
    B, C, H, W = images.shape
    _, N = gate_values.shape

    # 1) Teacher features
    target_features = model._encode_teacher(images)      # (B, N, D)

    # 2) Student tokens
    student_tokens = model.context_encoder(images)       # (B, N, D)

    # 3) Build context tokens with provided gates
    mask_token = model.mask_token.to(device)             # (1, D)
    mask_tokens = mask_token.expand(B, N, target_features.size(-1))

    r = gate_values.unsqueeze(-1)                        # (B, N, 1)
    one_minus_r = 1.0 - r                                # (B, N, 1)

    context_tokens = r * student_tokens + one_minus_r * mask_tokens  # (B, N, D)

    # 4) Predictor
    predictor_output = model.predictor(
        context_tokens,
        patchify=False,
        pos_embed=False,
    )
    predictor_output = F.layer_norm(
        predictor_output,
        (predictor_output.size(-1),),
    )

    # 5) Prediction loss on closed tokens (same as in model.forward)
    diff = predictor_output - target_features            # (B, N, D)
    sq_norm = (diff ** 2).sum(dim=-1)                    # (B, N)

    inv_gate = 1.0 - gate_values                         # (B, N)
    eps = 1e-6

    weighted_sum = (sq_norm * inv_gate).sum(dim=1)       # (B,)
    weight_norm = inv_gate.sum(dim=1) + eps              # (B,)

    pred_loss_per_sample = weighted_sum / weight_norm    # (B,)
    pred_loss = pred_loss_per_sample.mean()              # scalar
    return pred_loss


@torch.no_grad()
def compute_full_pred_loss_all_open(
    model: GatedIJEPA,
    images: torch.Tensor,
) -> torch.Tensor:
    """
    Compute a 'full' prediction loss where *no gating* is applied:
    the predictor sees all student tokens (no mask tokens), and we
    compute MSE over *all* tokens.

    This is different from L_pred in the model, which only looks at
    closed tokens. Here we treat it as a pure distillation loss:

        L_full = mean_{b,i} || LN(p(z^{(S)}_{b,i})) - LN(z^{(T)}_{b,i}) ||^2
    """
    device = images.device

    # 1) Teacher features
    target_features = model._encode_teacher(images)          # (B, N, D)

    # 2) Student tokens (used as context directly)
    student_tokens = model.context_encoder(images)           # (B, N, D)

    # 3) Predictor with full tokens, no gating
    predictor_output = model.predictor(
        student_tokens,
        patchify=False,
        pos_embed=False,
    )
    predictor_output = F.layer_norm(
        predictor_output,
        (predictor_output.size(-1),),
    )

    # 4) MSE over all tokens
    diff = predictor_output - target_features                # (B, N, D)
    sq_norm = (diff ** 2).sum(dim=-1)                        # (B, N)
    loss = sq_norm.mean()                                    # scalar
    return loss


@torch.no_grad()
def random_gate_ablation(
    model: GatedIJEPA,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 10,
) -> Dict[str, float]:
    """
    Compare prediction loss for:
      - learned gates r_learned (L_pred on closed tokens)
      - random Bernoulli gates r_rand with same per-sample open fraction
      - all-open (no masking at all, full-token distillation loss)

    Returns a dict with mean losses, stds, and mean open fraction
    (from the learned gates).
    """
    learned_losses = []
    random_losses = []
    all_open_losses = []
    open_fracs_all = []

    for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Random gate ablation")):
        if batch_idx >= max_batches:
            break

        images = images.to(device)
        B, C, H, W = images.shape

        # --- Learned gates ---
        student_tokens = model.context_encoder(images)
        log_alpha = model.gate_mlp(student_tokens).squeeze(-1)   # (B, N)
        gate_values, gate_probs = model.gate(log_alpha, training=False)  # (B, N)

        # Use gate_probs as expected open fraction per token
        open_frac_per_sample = gate_probs.mean(dim=1)            # (B,)
        open_fracs_all.append(open_frac_per_sample.cpu().numpy())

        # Learned gates pred loss (closed-token L_pred)
        learned_loss = compute_pred_loss_with_given_gates(
            model,
            images,
            gate_values,  # continuous in [0,1]
        )
        learned_losses.append(learned_loss.item())

        # --- Random gates with same per-sample open fraction ---
        B_, N = gate_probs.shape
        assert B_ == B
        q = open_frac_per_sample.unsqueeze(-1).expand(B, N)      # (B, N)
        rand_uniform = torch.rand_like(q)
        rand_gates = (rand_uniform < q).float()                  # (B, N) in {0,1}

        random_loss = compute_pred_loss_with_given_gates(
            model,
            images,
            rand_gates,
        )
        random_losses.append(random_loss.item())

        # --- All-open baseline: no masking at all, full-token loss ---
        all_open_loss = compute_full_pred_loss_all_open(model, images)
        all_open_losses.append(all_open_loss.item())

    if not learned_losses:
        raise RuntimeError("No batches processed in random_gate_ablation.")

    learned_losses = np.array(learned_losses)
    random_losses = np.array(random_losses)
    all_open_losses = np.array(all_open_losses)
    open_fracs_all = np.concatenate(open_fracs_all, axis=0)

    stats = {
        "learned_loss_mean": float(learned_losses.mean()),
        "learned_loss_std": float(learned_losses.std()),
        "random_loss_mean": float(random_losses.mean()),
        "random_loss_std": float(random_losses.std()),
        "all_open_loss_mean": float(all_open_losses.mean()),
        "all_open_loss_std": float(all_open_losses.std()),
        "open_frac_mean": float(open_fracs_all.mean()),
        "open_frac_std": float(open_fracs_all.std()),
        "num_batches": len(learned_losses),
        "learned_loss_vals": learned_losses,
        "random_loss_vals": random_losses,
        "all_open_loss_vals": all_open_losses,
    }
    return stats


def save_avg_map_heatmap(avg_map: np.ndarray, out_path: str) -> None:
    H, W = avg_map.shape
    plt.figure(figsize=(4, 4))
    im = plt.imshow(avg_map, origin="lower", aspect="equal")
    plt.title("Average gate probability map")
    plt.xlabel("Patch X")
    plt.ylabel("Patch Y")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="P(open)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_open_fracs_hist(open_fracs: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(5, 4))
    plt.hist(open_fracs, bins=30, alpha=0.8)
    plt.xlabel("Open fraction per sample")
    plt.ylabel("Count")
    plt.title("Distribution of open fractions")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_noise_vs_real_plot(stats: Dict[str, float], out_path: str) -> None:
    labels = ["Real", "Noise"]
    means = [stats["real_mean"], stats["noise_mean"]]
    stds = [stats["real_std"], stats["noise_std"]]

    x = np.arange(len(labels))

    plt.figure(figsize=(5, 4))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, labels)
    plt.ylabel("Open fraction")
    plt.title("Open fraction: real vs noise")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_ablation_losses_plot(stats: Dict[str, float], out_path: str) -> None:
    labels = ["Learned", "Random", "All-open"]
    means = [
        stats["learned_loss_mean"],
        stats["random_loss_mean"],
        stats["all_open_loss_mean"],
    ]
    stds = [
        stats["learned_loss_std"],
        stats["random_loss_std"],
        stats["all_open_loss_std"],
    ]

    x = np.arange(len(labels))

    plt.figure(figsize=(6, 4))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, labels)
    plt.ylabel("Prediction loss")
    plt.title("Random-gate ablation: prediction loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="GatedIJEPA gate analysis")

    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to trained GatedIJEPA checkpoint (.pth).",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data",
        help="Path where STL-10 dataset is stored.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "unlabeled"],
        help="STL-10 split to use for analysis.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for analysis.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Dataloader workers.",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=50,
        help="Max number of batches to use for some experiments.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./gate_analysis",
        help="Directory to save analysis artifacts (maps, stats, plots).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Build model and load checkpoint
    model = build_gated_model_from_ckpt(
        ckpt_path=args.ckpt_path,
        device=device,
    )

    # 2) Build dataset / dataloader
    transform = training_transforms((model.img_size, model.img_size))
    ds = STL10(
        args.dataset_path,
        split=args.split,
        transform=transform,
        download=True,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("\n[Experiment 1] Average gate map over dataset")
    avg_map, open_fracs = compute_gate_maps(
        model,
        loader,
        device=device,
        max_batches=args.num_batches,
    )
    print(f"  Mean open fraction over all samples: {open_fracs.mean():.4f}")
    print(f"  Std  open fraction over all samples: {open_fracs.std():.4f}")

    avg_map_path = os.path.join(args.out_dir, "avg_gate_map.pt")
    torch.save(torch.from_numpy(avg_map), avg_map_path)
    open_fracs_path = os.path.join(args.out_dir, "open_fracs.npy")
    np.save(open_fracs_path, open_fracs)
    print(f"  Saved average gate map tensor to: {avg_map_path}")
    print(f"  Saved open fractions to:          {open_fracs_path}")

    # Visualizations for Experiment 1
    avg_map_png = os.path.join(args.out_dir, "avg_gate_map.png")
    save_avg_map_heatmap(avg_map, avg_map_png)
    print(f"  Saved average gate map heatmap to: {avg_map_png}")

    open_fracs_png = os.path.join(args.out_dir, "open_fracs_hist.png")
    save_open_fracs_hist(open_fracs, open_fracs_png)
    print(f"  Saved open fraction histogram to:  {open_fracs_png}")

    # Simple positional summary: center vs corners
    H_p, W_p = avg_map.shape
    center = avg_map[H_p // 4: 3 * H_p // 4, W_p // 4: 3 * W_p // 4].mean()
    corners = np.mean([
        avg_map[0:H_p // 4, 0:W_p // 4],
        avg_map[0:H_p // 4, 3 * W_p // 4:W_p],
        avg_map[3 * H_p // 4:H_p, 0:W_p // 4],
        avg_map[3 * H_p // 4:H_p, 3 * W_p // 4:W_p],
    ])
    print(f"  Avg gate prob center region:  {center:.4f}")
    print(f"  Avg gate prob corner region:  {corners:.4f}")

    print("\n[Experiment 2] Noise vs real images (open fraction)")
    noise_stats = noise_vs_real_experiment(
        model,
        loader,
        device=device,
        max_batches=min(args.num_batches, 20),
    )
    print(
        f"  Real  open frac: mean={noise_stats['real_mean']:.4f}, "
        f"std={noise_stats['real_std']:.4f}"
    )
    print(
        f"  Noise open frac: mean={noise_stats['noise_mean']:.4f}, "
        f"std={noise_stats['noise_std']:.4f}"
    )

    noise_stats_path = os.path.join(args.out_dir, "noise_vs_real_stats.pt")
    # Strip arrays before saving stats to .pt for compactness
    noise_stats_to_save = {
        k: v for k, v in noise_stats.items() if k not in ("real_vals", "noise_vals")
    }
    torch.save(noise_stats_to_save, noise_stats_path)
    print(f"  Saved noise vs real stats to: {noise_stats_path}")

    noise_plot_path = os.path.join(args.out_dir, "noise_vs_real_open_frac.png")
    save_noise_vs_real_plot(noise_stats, noise_plot_path)
    print(f"  Saved noise vs real plot to:  {noise_plot_path}")

    print("\n[Experiment 3] Random-gate ablation (prediction loss)")
    ablation_stats = random_gate_ablation(
        model,
        loader,
        device=device,
        max_batches=min(args.num_batches, 20),
    )
    print(
        f"  Learned gates: loss={ablation_stats['learned_loss_mean']:.4f} "
        f"+/- {ablation_stats['learned_loss_std']:.4f}"
    )
    print(
        f"  Random  gates: loss={ablation_stats['random_loss_mean']:.4f} "
        f"+/- {ablation_stats['random_loss_std']:.4f}"
    )
    print(
        f"  All-open     : loss={ablation_stats['all_open_loss_mean']:.4f} "
        f"+/- {ablation_stats['all_open_loss_std']:.4f}"
    )
    print(
        f"  Open fraction (learned, mean +/- std): "
        f"{ablation_stats['open_frac_mean']:.4f} "
        f"+/- {ablation_stats['open_frac_std']:.4f}"
    )

    ablation_stats_path = os.path.join(args.out_dir, "random_gate_ablation_stats.pt")
    ablation_stats_to_save = {
        k: v for k, v in ablation_stats.items()
        if k not in ("learned_loss_vals", "random_loss_vals", "all_open_loss_vals")
    }
    torch.save(ablation_stats_to_save, ablation_stats_path)
    print(f"  Saved random-gate ablation stats to: {ablation_stats_path}")

    ablation_plot_path = os.path.join(args.out_dir, "random_gate_ablation_losses.png")
    save_ablation_losses_plot(ablation_stats, ablation_plot_path)
    print(f"  Saved random-gate ablation plot to: {ablation_plot_path}")

    print("\nDone. Inspect the saved tensors, arrays, and PNGs for deeper analysis.")


if __name__ == "__main__":
    main()
