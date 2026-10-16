"""Dataset inspection, preprocessing, and train/val/test splitting service."""

import os
import uuid
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from backend.app.core.config import settings
from backend.app.models.entities import DatasetRecord
from neuralforge_core.data import ArrayDataset, DataLoader


class DatasetService:
    @staticmethod
    def inspect_csv(file_path: str, preview_count: int = 10) -> Dict[str, Any]:
        """Inspect CSV file to compute column types, summary stats, and missing counts."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check file size limit
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > settings.MAX_UPLOAD_SIZE_MB:
            raise ValueError(f"File exceeds maximum upload limit of {settings.MAX_UPLOAD_SIZE_MB} MB")

        df = pd.read_csv(str(path))
        if df.empty:
            raise ValueError("The provided CSV file contains no data")

        row_count, col_count = df.shape
        columns_info = []

        for col in df.columns:
            s = df[col]
            missing_count = int(s.isna().sum())
            missing_pct = round(float(missing_count / row_count) * 100, 2)
            unique_count = int(s.nunique())
            dtype_str = str(s.dtype)

            sample_vals = [None if pd.isna(v) else (v if not isinstance(v, (np.integer, np.floating)) else float(v)) for v in s.head(5).tolist()]

            stat_dict = {
                "name": str(col),
                "dtype": dtype_str,
                "missing_count": missing_count,
                "missing_percentage": missing_pct,
                "unique_count": unique_count,
                "sample_values": sample_vals
            }

            if pd.api.types.is_numeric_dtype(s):
                valid_num = s.dropna()
                if not valid_num.empty:
                    stat_dict["min_val"] = round(float(valid_num.min()), 4)
                    stat_dict["max_val"] = round(float(valid_num.max()), 4)
                    stat_dict["mean_val"] = round(float(valid_num.mean()), 4)
                    stat_dict["std_val"] = round(float(valid_num.std()), 4)

            columns_info.append(stat_dict)

        # Preview rows (convert NaN to None for JSON serialization)
        preview_df = df.head(preview_count).copy()
        preview_rows = preview_df.replace({np.nan: None}).to_dict(orient="records")

        return {
            "row_count": row_count,
            "column_count": col_count,
            "columns": columns_info,
            "preview_rows": preview_rows
        }

    @staticmethod
    def prepare_data_loaders(
        dataset: DatasetRecord,
        batch_size: int = 32,
        seed: int = 42
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], Dict[str, Any]]:
        """Preprocess features and targets, preventing target leakage by fitting scalers only on train."""
        path = Path(dataset.file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        df = pd.read_csv(str(path))
        feature_cols = json.loads(dataset.feature_columns)
        target_col = dataset.target_column
        task_type = dataset.task_type

        if not feature_cols:
            raise ValueError("No feature columns selected for training")
        if not target_col or target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        # Missing values handling
        df = df.dropna(subset=[target_col]).copy()
        for col in feature_cols:
            if df[col].isna().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "missing")

        X_raw = df[feature_cols].copy()
        y_raw = df[target_col].copy()

        # Split ratios
        split_cfg = json.loads(dataset.split_config) if dataset.split_config else {}
        train_ratio = split_cfg.get("train_ratio", 0.7)
        val_ratio = split_cfg.get("val_ratio", 0.15)
        test_ratio = split_cfg.get("test_ratio", 0.15)
        split_seed = split_cfg.get("random_seed", seed)

        # First split into train and remainder
        val_plus_test = val_ratio + test_ratio
        if val_plus_test > 0.05:
            X_train, X_rem, y_train, y_rem = train_test_split(
                X_raw, y_raw,
                train_size=train_ratio,
                random_state=split_seed,
                shuffle=True
            )
            # Split remainder into val and test
            rel_val_ratio = val_ratio / val_plus_test
            if rel_val_ratio >= 0.99:
                X_val, y_val = X_rem, y_rem
                X_test, y_test = None, None
            elif rel_val_ratio <= 0.01:
                X_val, y_val = None, None
                X_test, y_test = X_rem, y_rem
            else:
                X_val, X_test, y_val, y_test = train_test_split(
                    X_rem, y_rem,
                    train_size=rel_val_ratio,
                    random_state=split_seed,
                    shuffle=True
                )
        else:
            X_train, y_train = X_raw, y_raw
            X_val, y_val = None, None
            X_test, y_test = None, None

        # Preprocessing: Fit scaler STRICTLY on train features only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val) if X_val is not None else None
        X_test_scaled = scaler.transform(X_test) if X_test is not None else None

        # Process target
        target_info = {}
        if task_type == "classification":
            encoder = LabelEncoder()
            y_train_encoded = encoder.fit_transform(y_train)
            y_val_encoded = encoder.transform(y_val) if y_val is not None else None
            y_test_encoded = encoder.transform(y_test) if y_test is not None else None
            target_info["classes"] = [str(c) for c in encoder.classes_]
            target_info["num_classes"] = len(encoder.classes_)
        else:
            y_train_encoded = y_train.to_numpy(dtype=np.float64).reshape(-1, 1)
            y_val_encoded = y_val.to_numpy(dtype=np.float64).reshape(-1, 1) if y_val is not None else None
            y_test_encoded = y_test.to_numpy(dtype=np.float64).reshape(-1, 1) if y_test is not None else None
            target_info["num_targets"] = 1

        train_dataset = ArrayDataset(X_train_scaled, y_train_encoded)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, seed=seed)

        val_loader = None
        if X_val_scaled is not None:
            val_dataset = ArrayDataset(X_val_scaled, y_val_encoded)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_loader = None
        if X_test_scaled is not None:
            test_dataset = ArrayDataset(X_test_scaled, y_test_encoded)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        metadata = {
            "input_dim": len(feature_cols),
            "output_dim": target_info.get("num_classes", target_info.get("num_targets", 1)),
            "target_info": target_info,
            "train_samples": len(X_train),
            "val_samples": len(X_val) if X_val is not None else 0,
            "test_samples": len(X_test) if X_test is not None else 0
        }

        return train_loader, val_loader, test_loader, metadata
