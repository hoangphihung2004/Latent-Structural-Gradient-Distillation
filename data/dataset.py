import os
from typing import Dict, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class RiceSeedDataset(Dataset):
    def __init__(self, data: pd.DataFrame, data_root: str, transform=None):
        self.data = data.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform

    def __getitem__(self, index: int):
        row = self.data.loc[index]
        label = torch.tensor(int(row["Label"]))

        path = os.path.join(
            self.data_root,
            row["Rice_Seed"],
            row["Encode_Label"],
            row["Path"],
        )
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label.long()

    def __len__(self) -> int:
        return len(self.data)


def load_splits(metadata_path: str, dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(metadata_path)
    df = df.loc[df["Rice_Seed"] == dataset_name].reset_index(drop=True)

    df_train = df.loc[df["Type"] == "Train"].copy().reset_index(drop=True)
    df_val = df.loc[df["Type"] == "Validation"].copy().reset_index(drop=True)
    df_test = df.loc[df["Type"] == "Test"].copy().reset_index(drop=True)

    return df_train, df_val, df_test


def build_dataloaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    data_root: str,
    transform_map: Dict,
    batch_size: int,
    num_workers: int = 2,
):
    train_dataset = RiceSeedDataset(df_train, data_root=data_root, transform=transform_map["Train"])
    valid_dataset = RiceSeedDataset(df_val, data_root=data_root, transform=transform_map["Validation"])
    test_dataset = RiceSeedDataset(df_test, data_root=data_root, transform=transform_map["Test"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {"Train": train_loader, "Validation": valid_loader}, test_loader

