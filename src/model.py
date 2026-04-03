import torch
import torch.nn as nn
import torchvision.models as models

class DepthNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18(pretrained=False)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        return self.decoder(x)


class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 64, 7, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 6, 3)
        )

    def forward(self, x):
        return self.net(x).mean([2,3])