import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class convNet(nn.Module):
    def __init__(self, num_classes):
        super(convNet, self).__init__()

        self.conv = models.resnet50(pretrained=True)
        for child in self.conv.children():
            for param in child.parameters():
                param.requires_grad = False

        num_ftrs = self.conv.fc.in_features
        self.conv.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.conv(x)
        return x
