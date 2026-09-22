
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class EfficientNetDropout(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        base = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.avgpool = base.avgpool
        num_feats = base.classifier[1].in_features
        self.fc = nn.Linear(num_feats, num_classes)

    def forward(self, x, dropout=0.0, get_embedding=False, reverse_grad=False):
        for i, block in enumerate(self.features):
            x = block(x)
            if i in (2, 4, 6, 8):
                x = F.dropout(x, p=dropout, training=True)
        x = self.avgpool(x)
        embedding = torch.flatten(x, 1)
        if reverse_grad:
            from models.ResNet import ReverseLayerF
            embedding = ReverseLayerF.apply(embedding)
        x = self.fc(embedding)
        if get_embedding:
            return x, embedding
        return x


def EfficientNetB0Dropout(num_classes=5, **kwargs):
    return EfficientNetDropout(num_classes=num_classes)
