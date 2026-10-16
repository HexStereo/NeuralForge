"""Generate reproducible sample datasets for NeuralForge demonstrations."""

from pathlib import Path
import numpy as np
import pandas as pd
from backend.app.core.config import settings


def generate_sample_dataset() -> Path:
    """Generate a reproducible multi-feature classification dataset."""
    sample_file = settings.DATA_DIR / "sample_sensor_faults.csv"
    if sample_file.exists():
        return sample_file

    np.random.seed(42)
    n_samples = 600

    # 4 continuous sensor features: temperature, vibration, pressure, current
    temp_normal = np.random.normal(65.0, 5.0, n_samples // 3)
    vib_normal = np.random.normal(12.0, 2.0, n_samples // 3)
    press_normal = np.random.normal(101.3, 3.0, n_samples // 3)
    curr_normal = np.random.normal(4.5, 0.5, n_samples // 3)
    labels_normal = ["Normal"] * (n_samples // 3)

    temp_overheat = np.random.normal(88.0, 6.0, n_samples // 3)
    vib_overheat = np.random.normal(15.0, 3.0, n_samples // 3)
    press_overheat = np.random.normal(108.0, 4.0, n_samples // 3)
    curr_overheat = np.random.normal(7.2, 0.8, n_samples // 3)
    labels_overheat = ["Overheating"] * (n_samples // 3)

    temp_imbalance = np.random.normal(68.0, 4.0, n_samples // 3)
    vib_imbalance = np.random.normal(28.0, 4.5, n_samples // 3)
    press_imbalance = np.random.normal(96.0, 3.5, n_samples // 3)
    curr_imbalance = np.random.normal(5.8, 0.7, n_samples // 3)
    labels_imbalance = ["Mechanical_Imbalance"] * (n_samples // 3)

    temp = np.concatenate([temp_normal, temp_overheat, temp_imbalance])
    vib = np.concatenate([vib_normal, vib_overheat, vib_imbalance])
    press = np.concatenate([press_normal, press_overheat, press_imbalance])
    curr = np.concatenate([curr_normal, curr_overheat, curr_imbalance])
    status = labels_normal + labels_overheat + labels_imbalance

    # Combine into dataframe and shuffle
    df = pd.DataFrame({
        "temperature_c": np.round(temp, 2),
        "vibration_rms": np.round(vib, 2),
        "pressure_kpa": np.round(press, 2),
        "motor_current_a": np.round(curr, 2),
        "fault_condition": status
    })

    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    sample_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(sample_file, index=False)
    return sample_file
