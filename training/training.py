from copy import deepcopy
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from .losses import GDLoss


def _maybe_data_parallel(model: nn.Module) -> nn.Module:
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return nn.DataParallel(model)
    return model


def _state_dict(model: nn.Module):
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def _load_state_dict(model: nn.Module, state_dict):
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


class DistillationTrainer:
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        criterion_ce: nn.Module,
        criterion_gd: nn.Module,
        optimizer_student: torch.optim.Optimizer,
        optimizer_teacher: torch.optim.Optimizer,
        device: torch.device,
        alpha: float = 1.0,
        delta: float = 0.7,
        early_stop: bool = True,
        patience: int = 20,
    ):
        self.student = _maybe_data_parallel(student).to(device)
        self.teacher = _maybe_data_parallel(teacher).to(device)
        self.criterion_ce = criterion_ce
        self.criterion_gd = criterion_gd
        self.optimizer_student = optimizer_student
        self.optimizer_teacher = optimizer_teacher
        self.device = device
        self.alpha = alpha
        self.delta = delta
        self.early_stop = early_stop
        self.patience = patience

        self.teacher.eval()

    @staticmethod
    def _compute_metrics(y_true, y_pred) -> Tuple[float, float, float]:
        if not y_true:
            return 0.0, 0.0, 0.0
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return precision, recall, f1

    def fit(
        self,
        data_loader: Dict[str, torch.utils.data.DataLoader],
        num_epochs: int,
    ) -> Tuple[Tuple[nn.Module, nn.Module], Tuple[pd.DataFrame, pd.DataFrame], float, Tuple[float, float], Tuple[int, int]]:
        best_student_wts = deepcopy(_state_dict(self.student))
        best_teacher_wts = deepcopy(_state_dict(self.teacher))
        best_val_loss_teacher, best_val_loss_student = float("inf"), float("inf")
        best_epoch_teacher, best_epoch_student = 0, 0
        wait = 0

        history_teacher = {
            "Train_Loss": [],
            "Train_Acc": [],
            "Train_Precision": [],
            "Train_Recall": [],
            "Train_F1": [],
            "Validation_Loss": [],
            "Validation_Acc": [],
            "Validation_Precision": [],
            "Validation_Recall": [],
            "Validation_F1": [],
            "Time": [],
        }

        history_student = {
            "Train_Loss": [],
            "Train_Acc": [],
            "Train_Precision": [],
            "Train_Recall": [],
            "Train_F1": [],
            "Validation_Loss": [],
            "Validation_Acc": [],
            "Validation_Precision": [],
            "Validation_Recall": [],
            "Validation_F1": [],
            "Time": [],
        }

        import time

        since_time = time.time()

        for epoch in range(1, num_epochs + 1):
            print("-----------------------------------------------------------------------")
            print(f"Epoch {epoch}/{num_epochs}")
            epoch_start = time.time()

            for phase in ["Train", "Validation"]:
                if phase == "Train":
                    self.teacher.train()
                    self.student.train()
                else:
                    self.teacher.eval()
                    self.student.eval()

                running_loss_teacher = 0.0
                running_corrects_teacher = 0
                running_loss_student = 0.0
                running_corrects_student = 0
                y_true_teacher = []
                y_pred_teacher = []
                y_true_student = []
                y_pred_student = []
                total_samples = 0

                for images, labels in tqdm(data_loader[phase], desc=f"{phase}"):
                    images, labels = images.to(self.device), labels.to(self.device)
                    batch_size = images.size(0)
                    total_samples += batch_size

                    with torch.set_grad_enabled(phase == "Train"):
                        logit_t, feat_t = self.teacher(images)
                        loss_teacher = self.criterion_ce(logit_t, labels)
                        pred_teacher = torch.argmax(logit_t, dim=1)

                        if phase == "Train":
                            self.optimizer_teacher.zero_grad()
                            loss_teacher.backward()
                            self.optimizer_teacher.step()

                        running_loss_teacher += loss_teacher.item() * batch_size
                        running_corrects_teacher += torch.sum(pred_teacher == labels).item()
                        y_true_teacher.extend(labels.detach().cpu().tolist())
                        y_pred_teacher.extend(pred_teacher.detach().cpu().tolist())

                    feat_t_detached = feat_t.detach()

                    with torch.set_grad_enabled(phase == "Train"):
                        logit_s, feat_s = self.student(images)
                        loss_ce = self.criterion_ce(logit_s, labels)
                        loss_gd = self.criterion_gd(feat_s, feat_t_detached)
                        total_loss_student = self.alpha * loss_ce + self.delta * loss_gd
                        pred_student = torch.argmax(logit_s, dim=1)

                        if phase == "Train":
                            self.optimizer_student.zero_grad()
                            total_loss_student.backward()
                            self.optimizer_student.step()

                        running_loss_student += total_loss_student.item() * batch_size
                        running_corrects_student += torch.sum(pred_student == labels).item()
                        y_true_student.extend(labels.detach().cpu().tolist())
                        y_pred_student.extend(pred_student.detach().cpu().tolist())

                if total_samples == 0:
                    epoch_loss_teacher = 0.0
                    epoch_acc_teacher = 0.0
                    epoch_precision_teacher = 0.0
                    epoch_recall_teacher = 0.0
                    epoch_f1_teacher = 0.0
                    epoch_loss_student = 0.0
                    epoch_acc_student = 0.0
                    epoch_precision_student = 0.0
                    epoch_recall_student = 0.0
                    epoch_f1_student = 0.0
                else:
                    epoch_loss_teacher = running_loss_teacher / total_samples
                    epoch_acc_teacher = running_corrects_teacher / total_samples
                    epoch_precision_teacher, epoch_recall_teacher, epoch_f1_teacher = self._compute_metrics(
                        y_true_teacher, y_pred_teacher
                    )

                    epoch_loss_student = running_loss_student / total_samples
                    epoch_acc_student = running_corrects_student / total_samples
                    epoch_precision_student, epoch_recall_student, epoch_f1_student = self._compute_metrics(
                        y_true_student, y_pred_student
                    )

                history_teacher[f"{phase}_Loss"].append(epoch_loss_teacher)
                history_teacher[f"{phase}_Acc"].append(epoch_acc_teacher)
                history_teacher[f"{phase}_Precision"].append(epoch_precision_teacher)
                history_teacher[f"{phase}_Recall"].append(epoch_recall_teacher)
                history_teacher[f"{phase}_F1"].append(epoch_f1_teacher)

                history_student[f"{phase}_Loss"].append(epoch_loss_student)
                history_student[f"{phase}_Acc"].append(epoch_acc_student)
                history_student[f"{phase}_Precision"].append(epoch_precision_student)
                history_student[f"{phase}_Recall"].append(epoch_recall_student)
                history_student[f"{phase}_F1"].append(epoch_f1_student)

                if phase == "Validation" and total_samples != 0:
                    if epoch_loss_student < best_val_loss_student:
                        best_val_loss_student = epoch_loss_student
                        best_student_wts = deepcopy(_state_dict(self.student))
                        best_epoch_student = epoch
                        wait = 0
                    else:
                        wait += 1

                    if epoch_loss_teacher < best_val_loss_teacher:
                        best_val_loss_teacher = epoch_loss_teacher
                        best_teacher_wts = deepcopy(_state_dict(self.teacher))
                        best_epoch_teacher = epoch

            epoch_duration = time.time() - epoch_start
            history_teacher["Time"].append(epoch_duration)
            history_student["Time"].append(epoch_duration)

            print("Teacher")
            print(
                f"Train Loss: {history_teacher['Train_Loss'][-1]:.4f}, "
                f"Train Acc: {history_teacher['Train_Acc'][-1]:.4f} | "
                f"Valid Loss: {history_teacher['Validation_Loss'][-1]:.4f}, "
                f"Valid Acc: {history_teacher['Validation_Acc'][-1]:.4f}"
            )

            print("Student")
            print(
                f"Train Loss: {history_student['Train_Loss'][-1]:.4f}, "
                f"Train Acc: {history_student['Train_Acc'][-1]:.4f} | "
                f"Valid Loss: {history_student['Validation_Loss'][-1]:.4f}, "
                f"Valid Acc: {history_student['Validation_Acc'][-1]:.4f}"
            )

            print(f"Epoch {epoch} finished in {epoch_duration:.2f}s")

            if self.early_stop and wait >= self.patience:
                print(f"Early stopping at epoch {epoch} (no improvement in {self.patience} epochs).")
                break

        print("-----------------------------------------------------------------------")
        time_elapsed = time.time() - since_time
        print(f"Training Completed in {time_elapsed}s")

        _load_state_dict(self.student, best_student_wts)
        _load_state_dict(self.teacher, best_teacher_wts)

        history = (pd.DataFrame(history_student), pd.DataFrame(history_teacher))
        best_val_loss = (best_val_loss_student, best_val_loss_teacher)
        best_epoch = (best_epoch_student, best_epoch_teacher)

        return (self.student, self.teacher), history, time_elapsed, best_val_loss, best_epoch


def train_model(
    data_loader: Dict[str, torch.utils.data.DataLoader],
    model,
    criterion,
    optimizer,
    num_epochs: int,
    device: torch.device,
    alpha: float = 1.0,
    delta: float = 0.7,
    early_stop: bool = True,
    patience: int = 20,
):
    student, teacher = model
    criterion_ce, criterion_gd = criterion
    optimizer_student, optimizer_teacher = optimizer

    trainer = DistillationTrainer(
        student=student,
        teacher=teacher,
        criterion_ce=criterion_ce,
        criterion_gd=criterion_gd,
        optimizer_student=optimizer_student,
        optimizer_teacher=optimizer_teacher,
        device=device,
        alpha=alpha,
        delta=delta,
        early_stop=early_stop,
        patience=patience,
    )
    return trainer.fit(data_loader=data_loader, num_epochs=num_epochs)
