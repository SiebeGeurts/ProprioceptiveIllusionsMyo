#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.spindle_FR_helper import load_coefficients
from utils.muscle_names import MUSCLE_NAMES

PARAM_KEYS = ["k_l", "k_v", "e_v", "k_a", "k_c", "max_rate", "frac_zero"]
DEFAULT_BASELINE_MODE = "power"
BASELINE_ROOT = os.path.join(os.path.dirname(__file__), "data", "spindle_coefficients")
EXTENDED_BASELINE_ROOT = os.path.join(os.path.dirname(__file__), "data", "extended_spindle_coefficients")


def infer_coeff_type(filepath):
    basename = os.path.basename(filepath).lower()
    if "i_a" in basename or "ia" in basename:
        return "i_a"
    if "ii" in basename:
        return "ii"
    raise ValueError(
        "Unable to infer coefficient type from filename. Use --type i_a or --type ii."
    )


def resolve_baseline_file(coeff_type, baseline_file, baseline_mode):
    if baseline_file is not None:
        return baseline_file

    candidate = os.path.join(BASELINE_ROOT, coeff_type, baseline_mode, "coefficients.csv")
    if os.path.exists(candidate):
        return candidate

    other_mode = "linear" if baseline_mode == "power" else "power"
    fallback = os.path.join(BASELINE_ROOT, coeff_type, other_mode, "coefficients.csv")
    if os.path.exists(fallback):
        print(
            f"Baseline file for mode '{baseline_mode}' not found. Falling back to '{other_mode}'.",
            file=sys.stderr,
        )
        return fallback

    raise FileNotFoundError(
        f"No baseline coefficient file found for type='{coeff_type}' in {BASELINE_ROOT}."
    )


def resolve_extended_baseline_file(coeff_type):
    candidate = os.path.join(EXTENDED_BASELINE_ROOT, coeff_type, "linear", "coefficients.csv")
    if os.path.exists(candidate):
        return candidate
    return None


def summarize_coefficients(coeffs):
    summary = {}
    for key in PARAM_KEYS:
        values = []
        for muscle_idx in sorted(coeffs.keys()):
            values.extend(coeffs[muscle_idx][key])
        values = np.array(values, dtype=float)
        summary[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def muscle_means(coeffs):
    muscle_summary = {}
    for muscle_idx in sorted(coeffs.keys()):
        muscle_summary[muscle_idx] = {
            key: float(np.mean(coeffs[muscle_idx][key])) for key in PARAM_KEYS
        }
    return muscle_summary


def flatten_coeff_values(coeffs, key):
    values = []
    for muscle_idx in sorted(coeffs.keys()):
        values.extend(coeffs[muscle_idx][key])
    return np.array(values, dtype=float)


def print_summary(label, summary):
    print(f"\n{label}")
    print("=" * len(label))
    for key in PARAM_KEYS:
        item = summary[key]
        print(
            f"{key:8s}: mean={item['mean']:.4f}, median={item['median']:.4f}, "
            f"std={item['std']:.4f}, min={item['min']:.4f}, max={item['max']:.4f}"
        )


def plot_baseline_vs_new_means(
    new_means,
    base_means,
    labels,
    coeff_type,
    out_dir,
    prefix,
):
    muscles = [MUSCLE_NAMES[m] for m in sorted(new_means.keys())]
    x = np.arange(len(muscles))
    width = 0.38

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), squeeze=False)
    axes = axes.flatten()
    for idx, key in enumerate(PARAM_KEYS):
        ax = axes[idx]
        new_vals = [new_means[muscle_idx][key] for muscle_idx in sorted(new_means.keys())]
        base_vals = [base_means[muscle_idx][key] for muscle_idx in sorted(base_means.keys())]
        ax.bar(x - width / 2, base_vals, width, label=labels[0], color="#4c72b0")
        ax.bar(x + width / 2, new_vals, width, label=labels[1], color="#dd8452")
        ax.set_title(f"{key} mean by muscle")
        ax.set_xticks(x)
        ax.set_xticklabels(muscles, rotation=90, fontsize=8)
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.legend()
    for idx in range(len(PARAM_KEYS), len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{prefix}_{coeff_type}_mean_comparison.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved mean comparison figure: {out_path}")


def plot_scatter_summary(
    new_means,
    base_means,
    coeff_type,
    out_dir,
    prefix,
):
    fig, axes = plt.subplots(2, 4, figsize=(18, 10), squeeze=False)
    axes = axes.flatten()
    for idx, key in enumerate(PARAM_KEYS):
        ax = axes[idx]
        base_vals = [base_means[m][key] for m in sorted(base_means.keys())]
        new_vals = [new_means[m][key] for m in sorted(new_means.keys())]
        ax.scatter(base_vals, new_vals, alpha=0.8)
        lims = [
            min(min(base_vals), min(new_vals)) * 0.95,
            max(max(base_vals), max(new_vals)) * 1.05,
        ]
        ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{key}: baseline vs new mean")
        ax.set_xlabel("baseline mean")
        ax.set_ylabel("new mean")
        ax.grid(alpha=0.2)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    axes[-1].axis("off")

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{prefix}_{coeff_type}_scatter_comparison.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter comparison figure: {out_path}")


def plot_histograms(
    new_vals,
    base_vals,
    coeff_type,
    out_dir,
    prefix,
    key,
    bins=25,
):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(base_vals, bins=bins, alpha=0.6, label="baseline", color="#4c72b0")
    ax.hist(new_vals, bins=bins, alpha=0.6, label="new", color="#dd8452")
    ax.set_title(f"Distribution of {key}")
    ax.set_xlabel(key)
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{prefix}_{coeff_type}_{key}_histogram.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histogram figure: {out_path}")


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize and compare new spindle coefficient sets against existing repository coefficients."
    )
    parser.add_argument(
        "new_file",
        help="Path to the newly generated coefficient CSV file (e.g. newspindledata/coefficients_i_a.csv).",
    )
    parser.add_argument(
        "--type",
        choices=["i_a", "ii"],
        help="Coefficient type when the filename cannot be inferred cleanly.",
    )
    parser.add_argument(
        "--baseline-file",
        help="Optional path to a baseline coefficient CSV file to compare against.",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=["power", "linear"],
        default=DEFAULT_BASELINE_MODE,
        help="Which pre-existing coefficient mode to compare to when no baseline file is provided.",
    )
    parser.add_argument(
        "--compare-extended",
        action="store_true",
        help="Also compare the new coefficients to the extended repository coefficients when available.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/spindle_comparison",
        help="Directory where comparison figures will be saved.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figures interactively after saving.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    coeff_type = args.type or infer_coeff_type(args.new_file)
    baseline_file = resolve_baseline_file(coeff_type, args.baseline_file, args.baseline_mode)

    new_coeffs = load_coefficients(args.new_file)
    baseline_coeffs = load_coefficients(baseline_file)
    extended_coeffs = None
    if args.compare_extended:
        extended_file = resolve_extended_baseline_file(coeff_type)
        if extended_file is not None:
            extended_coeffs = load_coefficients(extended_file)
            print(f"Loaded extended baseline coefficients from: {extended_file}")
        else:
            print(
                f"No extended baseline coefficient file found for type={coeff_type}. Skipping extended comparison.",
                file=sys.stderr,
            )

    output_dir = ensure_output_dir(args.output_dir)
    prefix = os.path.splitext(os.path.basename(args.new_file))[0]

    print(f"New coefficients: {args.new_file}")
    print(f"Baseline coefficients: {baseline_file}")
    if extended_coeffs is not None:
        print(f"Extended baseline coefficients: {extended_file}")

    new_summary = summarize_coefficients(new_coeffs)
    base_summary = summarize_coefficients(baseline_coeffs)
    ext_summary = summarize_coefficients(extended_coeffs) if extended_coeffs is not None else None

    print_summary("New coefficient summary", new_summary)
    print_summary("Baseline coefficient summary", base_summary)
    if ext_summary is not None:
        print_summary("Extended baseline summary", ext_summary)

    new_means = muscle_means(new_coeffs)
    base_means = muscle_means(baseline_coeffs)
    ext_means = muscle_means(extended_coeffs) if extended_coeffs is not None else None

    plot_baseline_vs_new_means(
        new_means,
        base_means,
        labels=("baseline", "new"),
        coeff_type=coeff_type,
        out_dir=output_dir,
        prefix=prefix,
    )
    plot_scatter_summary(
        new_means,
        base_means,
        coeff_type=coeff_type,
        out_dir=output_dir,
        prefix=prefix,
    )
    plot_histograms(
        flatten_coeff_values(new_coeffs, "max_rate"),
        flatten_coeff_values(baseline_coeffs, "max_rate"),
        coeff_type,
        output_dir,
        prefix,
        key="max_rate",
    )
    plot_histograms(
        flatten_coeff_values(new_coeffs, "frac_zero"),
        flatten_coeff_values(baseline_coeffs, "frac_zero"),
        coeff_type,
        output_dir,
        prefix,
        key="frac_zero",
    )

    if ext_means is not None:
        plot_baseline_vs_new_means(
            new_means,
            ext_means,
            labels=("extended", "new"),
            coeff_type=coeff_type,
            out_dir=output_dir,
            prefix=f"{prefix}_extended",
        )
        plot_scatter_summary(
            new_means,
            ext_means,
            coeff_type=coeff_type,
            out_dir=output_dir,
            prefix=f"{prefix}_extended",
        )

    if args.show:
        plt.show()

    print(f"\nFinished visualization. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
