from .losses import FDLoss, GDLoss, RDLoss
from .training import DistillationTrainer, train_model

KDLoss = RDLoss

__all__ = [
    "KDLoss",
    "RDLoss",
    "FDLoss",
    "GDLoss",
    "DistillationTrainer",
    "train_model",
]
