import torch
from torchvision import transforms


class ResNetNormed(torch.nn.Module):
    def __init__(self, model, mean=None, std=None):
        super().__init__()
        if mean is None or std is None:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2471, 0.2435, 0.2616]
        self.transfrom = transforms.Normalize(mean = mean, std = std)
        self.model=model
        # self.normalization_layer = NormalizationLayer(mean, std)

    def forward(self, x):
        out = self.transfrom(x)
        logits = self.model(out)
        return logits