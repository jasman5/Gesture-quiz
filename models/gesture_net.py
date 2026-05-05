import torch.nn as nn
import torchvision.models as models

class GestureNet(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3, norm='batch'):
        super().__init__()
        base = models.resnet18(pretrained=True)

        # Optionally replace BatchNorm with InstanceNorm
        if norm == 'instance':
            self._replace_bn(base, nn.InstanceNorm2d)

        in_feats = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Linear(in_feats, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        self.model = base

    def _replace_bn(self, module, norm_layer):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(module, name, norm_layer(child.num_features))
            else:
                self._replace_bn(child, norm_layer)

    def forward(self, x):
        return self.model(x)
