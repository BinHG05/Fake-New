"""
Fit scalar temperature scaling on validation logits and save calibration artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.paper_experiments.common import DEFAULT_CHECKPOINT, DEFAULT_GRAPH_DIR, DEFAULT_METADATA_PATH, load_binary_metadata


RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "paper_figures"
REPORT_PATH = RESULTS_DIR / "temperature_scaling_report.json"
FIGURE_PATH = FIGURES_DIR / "calibration_curve.png"


def lazy_import_torch():
    import torch
    from torch_geometric.data import Data

    return torch, Data


def load_val_graphs(graph_dir: Path = DEFAULT_GRAPH_DIR, metadata_path: Path = DEFAULT_METADATA_PATH):
    torch, Data = lazy_import_torch()
    torch.serialization.add_safe_globals([Data])
    metadata = load_binary_metadata(metadata_path)

    graphs = []
    for pt_file in sorted(graph_dir.glob("*.pt")):
        post_id = pt_file.stem
        meta = metadata.get(post_id)
        if not meta or meta["split"] != "val":
            continue
        graph = torch.load(pt_file, weights_only=False)
        graph.post_id = post_id
        graph.y = torch.tensor([meta["label"]], dtype=torch.long)
        if graph.edge_index.numel() == 0:
            graph.edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        graphs.append(graph)
    return graphs


def load_model(device: str = "cpu", checkpoint_path: Path = DEFAULT_CHECKPOINT):
    from scripts.paper_experiments.common import load_model

    return load_model(checkpoint_path=checkpoint_path, device=device)


def collect_logits(model, graphs, device: str = "cpu"):
    torch, _ = lazy_import_torch()
    from torch_geometric.loader import DataLoader

    loader = DataLoader(graphs, batch_size=32, shuffle=False)
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            logits_list.append(logits.detach().cpu())
            labels_list.append(batch.y.detach().cpu())

    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def fit_temperature(logits, labels):
    torch, _ = lazy_import_torch()

    temperature = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / temperature.clamp_min(1e-3), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp_min(1e-3).item())


def expected_calibration_error(confidences: np.ndarray, correctness: np.ndarray, bins: int = 10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    stats = []
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (confidences >= start) & (confidences < end if end < 1.0 else confidences <= end)
        if not np.any(mask):
            continue
        acc = float(correctness[mask].mean())
        conf = float(confidences[mask].mean())
        weight = float(mask.mean())
        ece += abs(acc - conf) * weight
        stats.append(
            {
                "bin_start": float(start),
                "bin_end": float(end),
                "count": int(mask.sum()),
                "accuracy": acc,
                "confidence": conf,
            }
        )
    return float(ece), stats


def calibration_points(logits, labels, temperature: float):
    torch, _ = lazy_import_torch()
    probs_before = torch.softmax(logits, dim=-1).numpy()
    probs_after = torch.softmax(logits / temperature, dim=-1).numpy()
    labels_np = labels.numpy()

    pred_before = probs_before.argmax(axis=1)
    pred_after = probs_after.argmax(axis=1)
    conf_before = probs_before.max(axis=1)
    conf_after = probs_after.max(axis=1)
    correct_before = (pred_before == labels_np).astype(float)
    correct_after = (pred_after == labels_np).astype(float)

    ece_before, bins_before = expected_calibration_error(conf_before, correct_before)
    ece_after, bins_after = expected_calibration_error(conf_after, correct_after)
    nll_before = float(torch.nn.functional.cross_entropy(logits, labels).item())
    nll_after = float(torch.nn.functional.cross_entropy(logits / temperature, labels).item())

    return {
        "before": {
            "ece": ece_before,
            "nll": nll_before,
            "bins": bins_before,
        },
        "after": {
            "ece": ece_after,
            "nll": nll_after,
            "bins": bins_after,
        },
    }


def plot_calibration(points: dict):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")

    for label, color in (("before", "#ef4444"), ("after", "#10b981")):
        bins = points[label]["bins"]
        if not bins:
            continue
        xs = [item["confidence"] for item in bins]
        ys = [item["accuracy"] for item in bins]
        plt.plot(xs, ys, marker="o", linewidth=2, color=color, label=f"{label.title()} (ECE={points[label]['ece']:.3f})")

    plt.xlabel("Mean confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability Diagram on Validation Set")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=200)
    plt.close()


def main():
    torch, _ = lazy_import_torch()
    checkpoint_path = DEFAULT_CHECKPOINT
    graphs = load_val_graphs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device=device, checkpoint_path=checkpoint_path)
    logits, labels = collect_logits(model, graphs, device=device)
    temperature = fit_temperature(logits.clone(), labels.clone())
    points = calibration_points(logits, labels, temperature)
    plot_calibration(points)

    calibration_path = checkpoint_path.with_name(f"{checkpoint_path.name}.calibration.json")
    calibration_payload = {
        "checkpoint": checkpoint_path.name,
        "temperature": temperature,
        "validation_samples": int(labels.size(0)),
        "before": {
            "ece": points["before"]["ece"],
            "nll": points["before"]["nll"],
        },
        "after": {
            "ece": points["after"]["ece"],
            "nll": points["after"]["nll"],
        },
        "figure": str(FIGURE_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    calibration_path.write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")

    print(json.dumps(calibration_payload, indent=2))


if __name__ == "__main__":
    main()
