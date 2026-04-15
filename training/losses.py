import torch
import torch.nn as nn
import torch.nn.functional as F


class RDLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super(RDLoss, self).__init__()

        self.temperature = temperature
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor):
        prob_student = F.log_softmax(student_logits / self.temperature, dim=1)
        prob_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        loss = self.criterion(prob_student, prob_teacher) * (self.temperature**2)
        return loss


class FDLoss(nn.Module):
    def __init__(self):
        super(FDLoss, self).__init__()

    def forward(self, student_feature: torch.Tensor, teacher_feature: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(student_feature, teacher_feature)


class GDLoss(nn.Module):
    def __init__(self):
        super(GDLoss, self).__init__()

    def forward(self, student_feature: torch.Tensor, teacher_feature: torch.Tensor):
        grad_student = torch.diff(student_feature)
        grad_teacher = torch.diff(teacher_feature)

        return F.mse_loss(grad_student, grad_teacher)

