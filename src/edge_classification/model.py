"""Model definitions for edge classification."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torchvision.models as models


class EdgeMultiLabelClassifier(nn.Module):
    """
    Lightweight multi-label edge classifier.

    The primary task predicts the four border labels:
    top, bottom, left, right.

    An optional auxiliary head predicts fragment piece type
    (interior / edge / corner) to stabilize training on small datasets.
    """

    def __init__(
        self,
        num_edge_labels: int = 4,
        piece_classes: int = 3,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        aux_piece_head: bool = True,
        freeze_backbone_fraction: float = 0.7,
    ) -> None:
        super().__init__()

        backbone = self._build_backbone(pretrained=pretrained)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = int(backbone.last_channel)
        self.aux_piece_head = aux_piece_head

        self._freeze_backbone_layers(freeze_backbone_fraction)

        self.edge_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.67),
            nn.Linear(256, num_edge_labels),
        )

        self.piece_head: Optional[nn.Sequential]
        if aux_piece_head:
            self.piece_head = nn.Sequential(
                nn.Dropout(dropout_rate * 0.67),
                nn.Linear(self.feature_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(128, piece_classes),
            )
        else:
            self.piece_head = None

    def _build_backbone(self, pretrained: bool) -> models.MobileNetV2:
        try:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            return models.mobilenet_v2(weights=weights)
        except Exception:
            # Fall back to random initialization when torchvision weights are
            # unavailable in an offline environment.
            return models.mobilenet_v2(weights=None)

    def _freeze_backbone_layers(self, freeze_backbone_fraction: float) -> None:
        freeze_backbone_fraction = min(max(float(freeze_backbone_fraction), 0.0), 1.0)
        feature_blocks = list(self.features.children())
        freeze_count = int(len(feature_blocks) * freeze_backbone_fraction)

        for block_index, block in enumerate(feature_blocks):
            requires_grad = block_index >= freeze_count
            for parameter in block.parameters():
                parameter.requires_grad = requires_grad

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = self.pool(features)
        return torch.flatten(features, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.extract_features(x)
        outputs: Dict[str, torch.Tensor] = {
            "edge_logits": self.edge_head(features),
        }
        if self.piece_head is not None:
            outputs["piece_logits"] = self.piece_head(features)
        return outputs


def load_edge_classifier(
    model_path: str,
    device: str = "cpu",
    pretrained: bool = False,
    aux_piece_head: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone_fraction: float = 0.0,
) -> EdgeMultiLabelClassifier:
    model = EdgeMultiLabelClassifier(
        pretrained=pretrained,
        aux_piece_head=aux_piece_head,
        dropout_rate=dropout_rate,
        freeze_backbone_fraction=freeze_backbone_fraction,
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
