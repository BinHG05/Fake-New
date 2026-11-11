"""
Evaluation metrics and plotting utilities for fake news classification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Computes comprehensive binary classification metrics."""
    targets = np.asarray(y_true, dtype=int)
    preds = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(targets, preds))
    f1 = float(f1_score(targets, preds, average="binary", zero_division=0))
    prec = float(precision_score(targets, preds, average="binary", zero_division=0))
    rec = float(recall_score(targets, preds, average="binary", zero_division=0))

    cm = confusion_matrix(targets, preds, labels=[0, 1]).tolist()

    res: dict[str, Any] = {
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "confusion_matrix": cm,
        "support": {
            "real": int((targets == 0).sum()),
            "fake": int((targets == 1).sum()),
            "total": int(len(targets)),
        },
    }

    if y_prob is not None:
        probs = np.asarray(y_prob, dtype=float)
        try:
            auc = float(roc_auc_score(targets, probs))
            res["roc_auc"] = round(auc, 4)
        except Exception:
            res["roc_auc"] = None

    return res


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    output_path: str | Path,
    labels: tuple[str, str] = ("REAL", "FAKE"),
) -> None:
    """Generates and saves a styled confusion matrix heatmap."""
    cm_arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    cax = ax.matshow(cm_arr, cmap=plt.cm.Blues, alpha=0.85)

    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            val = cm_arr[i, j]
            ax.text(
                j,
                i,
                f"{val:,}",
                ha="center",
                va="center",
                color="white" if val > cm_arr.max() / 2 else "black",
                fontsize=13,
                fontweight="bold",
            )

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Canonical Confusion Matrix", fontsize=12, fontweight="bold", pad=15)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close(fig)


def plot_reliability_curve(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    output_path: str | Path,
    n_bins: int = 10,
) -> None:
    """Plots probability calibration / reliability curve."""
    targets = np.asarray(y_true, dtype=int)
    probs = np.asarray(y_prob, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    true_fractions = []
    mean_predicted = []

    for i in range(n_bins):
        idx = (probs >= bins[i]) & (probs < bins[i + 1])
        if np.any(idx):
            true_fractions.append(np.mean(targets[idx]))
            mean_predicted.append(np.mean(probs[idx]))
        else:
            true_fractions.append(np.nan)
            mean_predicted.append(bin_centers[i])

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.plot(mean_predicted, true_fractions, "s-", color="#1f77b4", label="Model Calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Fraction of Positives (FAKE)", fontsize=10)
    ax.set_title("Reliability Curve", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close(fig)

