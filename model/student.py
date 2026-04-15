import torch
import torch.nn as nn
from torchvision import models


class Student(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(Student, self).__init__()

        self.model = models.shufflenet_v2_x0_5(
            weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
        )

        self.conv1 = self.model.conv1
        self.maxpool = self.model.maxpool
        self.stage2 = self.model.stage2
        self.stage3 = self.model.stage3
        self.stage4 = self.model.stage4

        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 768, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        feature = x.mean([2, 3])
        x = self.fc(feature)
        return x, feature

