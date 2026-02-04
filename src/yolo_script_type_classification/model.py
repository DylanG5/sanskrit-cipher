"""
Model module for script type classification
Efficient architecture using MobileNetV2 backbone
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EfficientScriptTypeClassifier(nn.Module):
    """
    Lightweight classifier using MobileNetV2 pretrained backbone
    Optimized for script type classification on limited data
    """
    
    def __init__(self, num_classes=5, pretrained=True, dropout_rate=0.3):
        super().__init__()
        
        # Load pretrained MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # Freeze early layers for transfer learning
        freeze_layers = len(list(self.backbone.parameters())) - 20
        for i, param in enumerate(self.backbone.parameters()):
            if i < freeze_layers:
                param.requires_grad = False
        
        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.67),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)
    
    def extract_features(self, x):
        """Extract features before classification head"""
        # Features from backbone excluding classifier
        features = self.backbone.features(x)
        features = torch.nn.functional.adaptive_avg_pool2d(features, 1)
        features = torch.flatten(features, 1)
        return features


def load_model(model_path, num_classes=5, device='cpu'):
    """Load a trained model from checkpoint"""
    model = EfficientScriptTypeClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
