"""
Standardized Inference Predictor for Fake News Detection.
Adheres strictly to the canonical data contract and portfolio requirements.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from src.fakenews.config.settings import get_project_root, load_config
from src.fakenews.graphs.cascade import CascadeGraphBuilder
from src.fakenews.labeling.labels import BINARY_LABEL_MAP, derive_binary_label
from src.fakenews.models.multimodal_gnn import MultimodalCascadeGNN

logger = logging.getLogger(__name__)


class FakeNewsPredictor:
    """
    Inference engine for multimodal cascade fake news classification.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        config_path: str | Path | None = None,
        device: str = "auto",
        threshold: float = 0.5,
        uncertain_threshold: float = 0.15,
    ) -> None:
        self.root = get_project_root()

        # Load config if available
        self.config: dict[str, Any] = {}
        try:
            self.config = load_config(config_path)
        except Exception:
            pass

        srv_cfg = self.config.get("serving", {})
        m_cfg = self.config.get("model", {})

        # Resolve checkpoint path
        if checkpoint_path is None:
            ckpt_str = srv_cfg.get(
                "checkpoint",
                "artifacts/checkpoints/multimodal_cascade_gnn_binary.pt",
            )
            self.checkpoint_path = (self.root / ckpt_str).resolve()
        else:
            p = Path(checkpoint_path)
            self.checkpoint_path = p if p.is_absolute() else (self.root / p).resolve()

        self.threshold = float(self.config.get("evaluation", {}).get("threshold", threshold))
        self.uncertain_threshold = float(srv_cfg.get("uncertain_threshold", uncertain_threshold))

        # Device resolution
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model hyperparameters
        self.text_dim = int(m_cfg.get("text_dim", 768))
        self.image_dim = int(m_cfg.get("image_dim", 1025))
        self.fusion_dim = int(m_cfg.get("fusion_dim", 512))
        self.hidden_dim = int(m_cfg.get("hidden_dim", 512))
        self.num_classes = int(m_cfg.get("num_classes", 2))
        self.num_gnn_layers = int(m_cfg.get("num_gnn_layers", 3))
        self.fusion_type = str(m_cfg.get("fusion", "cross_attention"))
        self.gnn_type = str(m_cfg.get("gnn", "gat"))

        self.model: Optional[MultimodalCascadeGNN] = None
        self.graph_builder = CascadeGraphBuilder(device=str(self.device))
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Canonical checkpoint not found: {self.checkpoint_path}. "
                "Ensure canonical model weights are placed in artifacts/checkpoints/."
            )

        self.model = MultimodalCascadeGNN(
            input_dim=self.text_dim + self.image_dim,
            text_dim=self.text_dim,
            image_dim=self.image_dim,
            fusion_dim=self.fusion_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_gnn_layers=self.num_gnn_layers,
            dropout=0.0,
            gnn_type=self.gnn_type,
            fusion_type=self.fusion_type,
        ).to(self.device)

        state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info("Loaded canonical model checkpoint: %s", self.checkpoint_path.name)

    def health(self) -> dict[str, Any]:
        """Returns runtime predictor status."""
        return {
            "status": "HEALTHY" if self.model is not None else "UNINITIALIZED",
            "checkpoint": str(self.checkpoint_path.name),
            "checkpoint_exists": self.checkpoint_path.exists(),
            "device": str(self.device),
            "threshold": self.threshold,
            "uncertain_threshold": self.uncertain_threshold,
            "model_architecture": {
                "type": "MultimodalCascadeGNN",
                "fusion": self.fusion_type,
                "gnn": self.gnn_type,
                "input_dim": self.text_dim + self.image_dim,
            },
        }

    def predict_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """
        Runs inference on a single record dictionary.
        Returns probabilities, prediction with UNCERTAIN band, and warning flags.
        """
        if self.model is None:
            raise RuntimeError("Predictor model is not loaded.")

        post_id = str(record.get("id", "sample")).strip()
        warnings: list[str] = []

        # Check media presence
        media = record.get("media") if isinstance(record.get("media"), dict) else {}
        image_path = record.get("image") or record.get("image_path") or media.get("processed_path")
        if not image_path:
            warnings.append("missing_image")

        cascade = record.get("cascade")
        if not cascade or len(cascade) == 0:
            warnings.append("missing_cascade_comments")

        # Build graph
        graph = self.graph_builder.build_graph(record, image_path=image_path)
        graph = graph.to(self.device)

        batch_idx = torch.zeros(graph.num_nodes, dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(graph.x, graph.edge_index, batch_idx)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        p_real = float(probs[0])
        p_fake = float(probs[1])

        # Confidence and margin relative to decision threshold
        margin = abs(p_fake - self.threshold)
        raw_pred = "FAKE" if p_fake >= self.threshold else "REAL"

        if margin < self.uncertain_threshold:
            decision = "UNCERTAIN"
        else:
            decision = raw_pred

        return {
            "id": post_id,
            "prediction": decision,
            "raw_prediction": raw_pred,
            "probability_fake": round(p_fake, 4),
            "probability_real": round(p_real, 4),
            "confidence": round(max(p_fake, p_real), 4),
            "margin_to_threshold": round(margin, 4),
            "decision_threshold": self.threshold,
            "warnings": warnings,
            "metadata": {
                "graph_nodes": int(graph.num_nodes),
                "graph_edges": int(graph.edge_index.size(1)),
                "has_image": bool(image_path),
                "cascade_length": len(cascade) if isinstance(cascade, list) else 0,
            },
        }

    def predict_batch(self, jsonl_path: str | Path) -> dict[str, Any]:
        """
        Runs batch prediction over a JSONL file.
        Returns list of individual predictions and batch-level summary.
        """
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Batch file not found: {path}")

        predictions: list[dict[str, Any]] = []
        counts = {"REAL": 0, "FAKE": 0, "UNCERTAIN": 0}
        confidences: list[float] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                res = self.predict_record(record)
                predictions.append(res)
                counts[res["prediction"]] += 1
                confidences.append(res["confidence"])

        total = len(predictions)
        avg_conf = float(sum(confidences) / total) if total > 0 else 0.0

        return {
            "total_records": total,
            "prediction_counts": counts,
            "average_confidence": round(avg_conf, 4),
            "predictions": predictions,
        }

