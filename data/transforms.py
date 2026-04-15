import numpy as np
from torchvision import transforms


def build_transforms() -> dict:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    return {
        "Train": transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        "Validation": transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        "Test": transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    }

