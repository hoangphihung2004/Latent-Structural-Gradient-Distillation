from dataclasses import dataclass
import os


@dataclass
class TrainConfig:
    dataset_name: str = "Q5"
    batch_size: int = 64
    learning_rate: float = 1e-4
    epochs: int = 100
    alpha: float = 1.0
    delta: float = 0.4
    num_workers: int = 2
    early_stop: bool = True
    patience: int = 20

    data_root: str = "Rice_Seed_Variety"
    output_dir: str = "Results"

    @property
    def metadata_path(self) -> str:
        return os.path.join(self.data_root, "metadata.csv")
