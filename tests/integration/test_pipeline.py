"""Offline integration test for canonical serving components."""

from __future__ import annotations

import torch
import yaml
import tempfile
from pathlib import Path
from torch_geometric.data import Data

from src.fakenews.models.multimodal_gnn import MultimodalCascadeGNN
from src.fakenews.serving.predictor import FakeNewsPredictor


class DeterministicGraphBuilder:
    """Avoids network/model downloads while exercising the real predictor."""

    def build_graph(self, item, image_path=None):
        return Data(
            x=torch.ones((1, 7), dtype=torch.float32),
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),
            post_id=str(item.get("id", "sample")),
            num_nodes=1,
        )


def test_predictor_integration_is_offline():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        temp_path = Path(temp_dir)
        config_path = temp_path / "test_config.yaml"
        checkpoint_path = temp_path / "tiny_model.pt"
        config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "text_dim": 4,
                    "image_dim": 3,
                    "fusion_dim": 8,
                    "hidden_dim": 8,
                    "num_classes": 2,
                    "num_gnn_layers": 1,
                    "fusion": "cross_attention",
                    "gnn": "gat",
                },
                "evaluation": {"threshold": 0.5},
                "serving": {"checkpoint": str(checkpoint_path), "uncertain_threshold": 0.15},
            }
        ),
            encoding="utf-8",
        )
        model = MultimodalCascadeGNN(
            input_dim=7,
            text_dim=4,
            image_dim=3,
            fusion_dim=8,
            hidden_dim=8,
            num_classes=2,
            num_gnn_layers=1,
            dropout=0.0,
        )
        torch.save(model.state_dict(), checkpoint_path)

        predictor = FakeNewsPredictor(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            graph_builder=DeterministicGraphBuilder(),
            device="cpu",
        )
        result = predictor.predict_record(
            {"id": "integration_test_item", "clean_text": "offline fixture", "cascade": []}
        )

        assert result["id"] == "integration_test_item"
        assert result["prediction"] in {"REAL", "FAKE", "UNCERTAIN"}
        assert 0.0 <= result["probability_fake"] <= 1.0
        assert "missing_image" in result["warnings"]
        assert "missing_cascade_comments" in result["warnings"]
