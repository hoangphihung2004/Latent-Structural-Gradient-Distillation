import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from configs import TrainConfig
from data import build_dataloaders, build_transforms, load_splits
from evaluation import evaluate_model
from model import Student, Teacher
from .losses import GDLoss
from .training import DistillationTrainer
from utils import (
    ensure_dir,
    save_classification_report,
    save_confusion_matrix,
    save_results_csv,
    save_model_weights,
    zip_dir,
)


class DistillationPipeline:
    def __init__(self, config: TrainConfig | None = None):
        self.config = config or TrainConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path = self.config.output_dir
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.dataloaders = None
        self.test_loader = None
        self.num_classes = 2
        self.student = None
        self.teacher = None
        self.results = {
            "Name_Dataset": self.config.dataset_name,
            "Method": "Distillation",
            "Batch_Size": self.config.batch_size,
            "Lr": self.config.learning_rate,
            "Epochs": self.config.epochs,
            "Optimizer": "Adam",
            "Alpha": self.config.alpha,
            "Delta": self.config.delta,
        }

    def run(self):
        ensure_dir(self.path)
        self._prepare_data()
        self._build_models()
        history_student, history_teacher = self._train_models()
        self._save_training_artifacts(history_student, history_teacher)
        self._evaluate_and_save()
        self._save_results_and_archive()
        print(f"Saved artifacts in: {self.path}")
        return self.results

    def _prepare_data(self):
        self.df_train, self.df_val, self.df_test = load_splits(self.config.metadata_path, self.config.dataset_name)
        transforms_map = build_transforms()
        self.dataloaders, self.test_loader = build_dataloaders(
            self.df_train,
            self.df_val,
            self.df_test,
            data_root=self.config.data_root,
            transform_map=transforms_map,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

        labels = pd.concat([self.df_train["Label"], self.df_val["Label"], self.df_test["Label"]], ignore_index=True)
        self.num_classes = len(pd.unique(labels))
        print(f"Using device: {self.device}")

    def _build_models(self):
        self.student = Student(num_classes=self.num_classes)
        self.teacher = Teacher(num_classes=self.num_classes)

    def _train_models(self):
        criterion_ce = nn.CrossEntropyLoss()
        criterion_gd = GDLoss()
        optimizer_student = torch.optim.Adam(self.student.parameters(), lr=self.config.learning_rate)
        optimizer_teacher = torch.optim.Adam(self.teacher.parameters(), lr=self.config.learning_rate)

        trainer = DistillationTrainer(
            student=self.student,
            teacher=self.teacher,
            criterion_ce=criterion_ce,
            criterion_gd=criterion_gd,
            optimizer_student=optimizer_student,
            optimizer_teacher=optimizer_teacher,
            device=self.device,
            alpha=self.config.alpha,
            delta=self.config.delta,
            early_stop=self.config.early_stop,
            patience=self.config.patience,
        )

        model, history, time_elapsed, best_val_loss, best_epoch = trainer.fit(
            data_loader=self.dataloaders,
            num_epochs=self.config.epochs,
        )
        self.student, self.teacher = model
        self.results["Best_val_loss_student"] = best_val_loss[0]
        self.results["Best_val_loss_teacher"] = best_val_loss[1]
        self.results["Best_epoch_student"] = best_epoch[0]
        self.results["Best_epoch_teacher"] = best_epoch[1]
        self.results["time_elapsed"] = time_elapsed
        return history

    def _plot_history(self, history: pd.DataFrame, title_prefix: str, output_path: str) -> None:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

        axs[0].plot(history["Train_Acc"], label="Train Accuracy")
        axs[0].plot(history["Validation_Acc"], label="Validation Accuracy")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Accuracy")
        axs[0].set_title(f"{title_prefix} Training and Validation Accuracy")
        axs[0].legend()

        axs[1].plot(history["Train_Loss"], label="Train Loss")
        axs[1].plot(history["Validation_Loss"], label="Validation Loss")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Loss")
        axs[1].set_title(f"{title_prefix} Training and Validation Loss")
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)

    def _save_training_artifacts(self, history_student: pd.DataFrame, history_teacher: pd.DataFrame) -> None:
        history_student.to_csv(os.path.join(self.path, "history_student.csv"), index=False)
        history_teacher.to_csv(os.path.join(self.path, "history_teacher.csv"), index=False)
        self._plot_history(history_teacher, "Teacher", os.path.join(self.path, "plot_loss_acc_teacher.png"))
        self._plot_history(history_student, "Student", os.path.join(self.path, "plot_loss_acc_student.png"))

    def _evaluate_single_model(self, model, prefix: str):
        metrics, report, cm, _, _ = evaluate_model(model, self.test_loader, self.device)

        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Precision: {metrics['precision']}")
        print(f"Recall: {metrics['recall']}")
        print(f"F1-Score: {metrics['f1']}")
        print(report)

        self.results[f"Accuracy_{prefix}"] = metrics["accuracy"]
        self.results[f"Precision_{prefix}"] = metrics["precision"]
        self.results[f"Recall_{prefix}"] = metrics["recall"]
        self.results[f"F1-Score_{prefix}"] = metrics["f1"]

        save_classification_report(report, self.path, f"classification_report_{prefix.lower()}.txt")
        save_confusion_matrix(cm, self.path, f"confusion_matrix_{prefix.lower()}.png")

    def _evaluate_and_save(self) -> None:
        self._evaluate_single_model(self.teacher, "Teacher")
        self._evaluate_single_model(self.student, "Student")

    def _save_results_and_archive(self) -> None:
        save_results_csv(self.results, self.path, "result.csv")
        save_model_weights(self.student, self.path, "student_weights.pth")
        save_model_weights(self.teacher, self.path, "teacher_weights.pth")
        zip_dir(self.path, f"{self.path}.zip")

