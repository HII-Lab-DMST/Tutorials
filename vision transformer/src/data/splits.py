"""
Generic patient-level splitting utilities for any labelled DataFrame.

No dataset-specific logic lives here. All functions take explicit column name
arguments so they work with any DataFrame schema.

Typical usage
-------------
    from src.data.splits import patient_split, kfold_patient_split

    train_df, val_df = patient_split(
        df, patient_col="Patient_ID", label_col="Label"
    )

    for fold, train_df, val_df in kfold_patient_split(
        df, patient_col="Case", label_col="Pathology"
    ):
        ...
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def patient_split(
    df: pd.DataFrame,
    patient_col: str,
    label_col: str,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Patient-level stratified train/val split.

    Stratification uses each patient's majority label, so class balance is
    preserved even when one patient contributes multiple images.

    Parameters
    ----------
    df : pd.DataFrame
        Labels DataFrame. Rows with NaN in ``label_col`` must be removed first.
    patient_col : str
        Column containing patient identifiers (e.g. ``"Patient_ID"``, ``"Case"``).
    label_col : str
        Column containing integer or binary labels.
    val_size : float
        Fraction of *patients* (not images) assigned to validation.
    random_state : int

    Returns
    -------
    train_df, val_df : pd.DataFrame
        Row-subsets of ``df`` with reset indices. No patient appears in both.
    """
    patient_label = (
        df.groupby(patient_col)[label_col]
        .agg(lambda s: s.mode().iloc[-1])   # majority vote; ties go to last mode
    )
    patient_ids = patient_label.index.to_numpy()
    patient_labels = patient_label.values

    train_patients, val_patients = train_test_split(
        patient_ids,
        test_size=val_size,
        stratify=patient_labels,
        random_state=random_state,
    )

    train_df = df[df[patient_col].isin(train_patients)].reset_index(drop=True)
    val_df   = df[df[patient_col].isin(val_patients)].reset_index(drop=True)

    return train_df, val_df


def kfold_patient_split(
    df: pd.DataFrame,
    patient_col: str,
    label_col: str,
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    Patient-level stratified k-fold split.

    Stratification operates at the patient level (majority label per patient),
    so no patient leaks across the fold boundary.

    Parameters
    ----------
    df : pd.DataFrame
        Labels DataFrame. Rows with NaN in ``label_col`` must be removed first.
    patient_col : str
        Column containing patient identifiers.
    label_col : str
        Column containing integer or binary labels.
    n_splits : int
        Number of folds (default 5).
    random_state : int

    Yields
    ------
    fold : int
        Zero-indexed fold number.
    train_df : pd.DataFrame
    val_df : pd.DataFrame
        Each with reset indices; no patient overlap between train and val.

    Example
    -------
        for fold, train_df, val_df in kfold_patient_split(df, "Patient_ID", "Label"):
            train_ds = MyDataset(train_df, root_dir, transform=train_tf)
            val_ds   = MyDataset(val_df,   root_dir, transform=val_tf)
            train_loader, val_loader = build_loaders(train_ds, val_ds)
    """
    patient_label = (
        df.groupby(patient_col)[label_col]
        .agg(lambda s: s.mode().iloc[-1])
    )
    patient_ids    = patient_label.index.to_numpy()
    patient_labels = patient_label.values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(skf.split(patient_ids, patient_labels)):
        train_patients = set(patient_ids[train_idx])
        val_patients   = set(patient_ids[val_idx])

        train_df = df[df[patient_col].isin(train_patients)].reset_index(drop=True)
        val_df   = df[df[patient_col].isin(val_patients)].reset_index(drop=True)

        yield fold, train_df, val_df
