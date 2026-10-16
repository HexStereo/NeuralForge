"""Evaluation metrics for classification and regression models."""

from typing import Dict, Any, Optional
import numpy as np


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute classification accuracy."""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        preds = np.argmax(y_pred, axis=-1)
    elif y_pred.ndim > 1 and y_pred.shape[1] == 1:
        preds = (y_pred.flatten() >= 0.5).astype(int)
    else:
        preds = y_pred

    if y_true.ndim > 1 and y_true.shape[1] > 1:
        trues = np.argmax(y_true, axis=-1)
    else:
        trues = y_true.flatten()

    return float(np.mean(preds == trues))


def mean_absolute_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return float(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2))


def root_mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_pred, y_true)))


def r2_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Coefficient of Determination R^2."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """Compute confusion matrix where rows are actual classes and columns are predicted classes."""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if y_pred.ndim > 1:
        preds = np.argmax(y_pred, axis=-1)
    else:
        preds = y_pred.astype(int)

    if y_true.ndim > 1:
        trues = np.argmax(y_true, axis=-1)
    else:
        trues = y_true.astype(int)

    if num_classes is None:
        num_classes = max(int(np.max(preds)), int(np.max(trues))) + 1

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(trues, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def classification_report(y_pred: np.ndarray, y_true: np.ndarray, num_classes: Optional[int] = None) -> Dict[str, Any]:
    """Calculate per-class precision, recall, F1, and aggregate statistics."""
    cm = confusion_matrix(y_pred, y_true, num_classes=num_classes)
    n_classes = cm.shape[0]

    classes_report = {}
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for c in range(n_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        support = int(np.sum(cm[c, :]))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        classes_report[str(c)] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    total_support = sum(supports)
    macro_p = float(np.mean(precisions)) if precisions else 0.0
    macro_r = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    weighted_p = float(sum(p * s for p, s in zip(precisions, supports)) / total_support) if total_support > 0 else 0.0
    weighted_r = float(sum(r * s for r, s in zip(recalls, supports)) / total_support) if total_support > 0 else 0.0
    weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / total_support) if total_support > 0 else 0.0

    overall_acc = float(np.trace(cm) / total_support) if total_support > 0 else 0.0

    return {
        "accuracy": round(overall_acc, 4),
        "classes": classes_report,
        "macro_avg": {
            "precision": round(macro_p, 4),
            "recall": round(macro_r, 4),
            "f1_score": round(macro_f1, 4)
        },
        "weighted_avg": {
            "precision": round(weighted_p, 4),
            "recall": round(weighted_r, 4),
            "f1_score": round(weighted_f1, 4)
        },
        "confusion_matrix": cm.tolist()
    }
