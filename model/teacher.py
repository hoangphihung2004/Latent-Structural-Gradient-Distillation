import torch
import torch.nn as nn
from torchvision import models


class Teacher(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

        in_dim = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.vit._process_input(x)
        n = x.shape[0]

        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit.encoder(x)
        cls_feat = x[:, 0]
        logits = self.vit.heads(cls_feat)

        return logits, cls_feat

