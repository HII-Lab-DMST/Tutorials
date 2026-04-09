"""
Dataset for BUS-BRA (Breast Ultrasound B-mode and Automated 3D data).

Directory layout expected under root_dir:
    bus_data.csv    — columns include: ID, Case, Pathology
    Images/         — image files named {ID}.*

The class accepts a pre-filtered DataFrame. Load the CSV and run patient_split
before constructing datasets. Splitting logic lives in src/data/splits.py.

Note on labels
--------------
Pathology is a string column ("malignant" / "benign"). This class encodes it as:
    1  →  malignant
    0  →  benign (or anything else)

Typical usage
-------------
    import pandas as pd
    from src.data.splits import patient_split
    from src.data.loaders import build_loaders
    from src.data.datasets.busbra import BUSBRADataset

    ROOT = "data/raw/BUSBRA_Dataset/BUSBRA/BUSBRA"
    df = pd.read_csv(f"{ROOT}/bus_data.csv")

    train_df, val_df = patient_split(df, patient_col="Case", label_col="Pathology")

    train_ds = BUSBRADataset(train_df, ROOT, transform=train_tf)
    val_ds   = BUSBRADataset(val_df,   ROOT, transform=val_tf)

    train_loader, val_loader = build_loaders(train_ds, val_ds, batch_size=32)
"""

import glob
import os
import warnings

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class BUSBRADataset(Dataset):
    """
    Dataset for BUS-BRA.

    Parameters
    ----------
    df : pd.DataFrame
        Subset of bus_data.csv. Required columns: ``ID``, ``Pathology``.
    root_dir : str
        Path to the BUS-BRA root directory (parent of the Images/ folder).
    transform : callable, optional
        Applied to the PIL.Image before returning. Should return a tensor.
    """

    def __init__(self, df: pd.DataFrame, root_dir: str, transform=None):
        self.root_dir  = root_dir
        self.transform = transform

        df = df.reset_index(drop=True)

        paths, keep = [], []
        for i in range(len(df)):
            path = self._find_image(df.iloc[i], root_dir)
            if path is not None:
                paths.append(path)
                keep.append(i)
            else:
                warnings.warn(
                    f"Image not found for ID='{df.iloc[i]['ID']}' — skipping row.",
                    UserWarning,
                    stacklevel=2,
                )

        self.df     = df.iloc[keep].reset_index(drop=True)
        self._paths = paths

    # ------------------------------------------------------------------

    @staticmethod
    def _find_image(row: pd.Series, root_dir: str) -> str | None:
        matches = glob.glob(os.path.join(root_dir, "Images", f"{row['ID']}.*"))
        return matches[0] if matches else None

    @staticmethod
    def _encode_label(pathology: str) -> int:
        return 1 if str(pathology).strip().lower() == "malignant" else 0

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self._paths[idx]).convert("RGB")
        label = self._encode_label(self.df.iloc[idx]["Pathology"])

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

    def label_array(self) -> np.ndarray:
        """Integer label for every sample. Used by loaders.get_weighted_sampler."""
        return self.df["Pathology"].apply(self._encode_label).to_numpy()
