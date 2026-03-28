"""ResNet-50 classifier for GI disease classification.

Wraps torchvision's ResNet-50 with a configurable classification head
and utilities for transfer-learning workflows (freeze/unfreeze backbone,
load pretrained weights).
"""

import logging

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

logger = logging.getLogger(__name__)


class ResNet50Classifier(nn.Module):
    """ResNet-50 with a replaceable classification head.

    The final fully-connected layer is replaced with a new linear layer
    sized to ``num_classes``.  An optional dropout layer sits in front of
    the classifier for regularisation.

    Args:
        num_classes: Number of output classes.
        pretrained: If True, load ImageNet-pretrained backbone weights.
        dropout: Dropout probability before the final linear layer.

    Example:
        >>> model = ResNet50Classifier(num_classes=8, pretrained=True)
        >>> x = torch.randn(4, 3, 224, 224)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([4, 8])
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        logger.info(
            "ResNet50Classifier: num_classes=%d, pretrained=%s, dropout=%.2f",
            num_classes, pretrained, dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_pretrained_resnet50(
    num_classes: int,
    pretrained: bool = True,
) -> ResNet50Classifier:
    """Create a ResNet-50 classifier, optionally with ImageNet weights.

    This is a convenience factory. For more control (dropout, device
    placement) instantiate ``ResNet50Classifier`` directly.

    Args:
        num_classes: Number of output classes.
        pretrained: Load ImageNet-pretrained backbone weights.

    Returns:
        Initialised ``ResNet50Classifier``.

    Example:
        >>> model = load_pretrained_resnet50(num_classes=8)
    """
    return ResNet50Classifier(num_classes=num_classes, pretrained=pretrained)


def freeze_backbone(model: ResNet50Classifier, freeze: bool = True) -> None:
    """Freeze or unfreeze all backbone parameters except the classification head.

    When frozen only the final FC layer (and its dropout) receives
    gradients, which is useful for the early phase of transfer learning.

    Args:
        model: A ``ResNet50Classifier`` instance.
        freeze: If True, freeze the backbone. If False, unfreeze it.

    Example:
        >>> model = load_pretrained_resnet50(8)
        >>> freeze_backbone(model, freeze=True)   # only train head
        >>> freeze_backbone(model, freeze=False)   # fine-tune everything
    """
    for name, param in model.backbone.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = not freeze

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Backbone %s — trainable params: %s", "frozen" if freeze else "unfrozen", f"{trainable:,}")