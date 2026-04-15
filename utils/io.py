import os
import zipfile
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_history_csv(history: Dict[str, list], output_dir: str, filename: str = "history.csv") -> str:
    ensure_dir(output_dir)
    path = os.path.join(output_dir, filename)
    pd.DataFrame(history).to_csv(path, index=False)
    return path


def save_results_csv(results: Dict[str, object], output_dir: str, filename: str = "results.csv") -> str:
    ensure_dir(output_dir)
    path = os.path.join(output_dir, filename)
    pd.DataFrame([results]).to_csv(path, index=False)
    return path


def save_classification_report(report: str, output_dir: str, filename: str = "classification_report.txt") -> str:
    ensure_dir(output_dir)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    return path


def save_confusion_matrix(cm, output_dir: str, filename: str = "confusion_matrix.png") -> str:
    ensure_dir(output_dir)
    path = os.path.join(output_dir, filename)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path


def save_training_curves(history: Dict[str, list], output_dir: str) -> str:
    ensure_dir(output_dir)
    path = os.path.join(output_dir, "training_curves.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history["train_loss"], label="train")
    axes[0, 0].plot(history["val_loss"], label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(history["train_acc"], label="train")
    axes[0, 1].plot(history["val_acc"], label="val")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].legend()

    axes[1, 0].plot(history["train_precision"], label="train")
    axes[1, 0].plot(history["val_precision"], label="val")
    axes[1, 0].set_title("Precision")
    axes[1, 0].legend()

    axes[1, 1].plot(history["train_f1_score"], label="train")
    axes[1, 1].plot(history["val_f1_score"], label="val")
    axes[1, 1].set_title("F1-score")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return path


def save_model_weights(model: torch.nn.Module, output_dir: str, filename: str = "student_weights.pth") -> str:
    ensure_dir(output_dir)
    path = os.path.join(output_dir, filename)
    torch.save(model.state_dict(), path)
    return path


def zip_dir(input_dir: str, output_zip: str) -> str:
    ensure_dir(os.path.dirname(output_zip) or ".")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(input_dir):
            for file_name in files:
                full_path = os.path.join(root, file_name)
                arcname = os.path.relpath(full_path, input_dir)
                zipf.write(full_path, arcname)
    return output_zip

