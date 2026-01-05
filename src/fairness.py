"""
Fairness metrics and evaluation utilities.
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def demographic_parity(
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray
) -> float:
    """
    Calculate demographic parity difference.
    
    Demographic parity requires that the proportion of positive predictions
    be the same across different groups.
    
    Args:
        y_pred: Predicted labels
        sensitive_attr: Sensitive attribute values
        
    Returns:
        Demographic parity difference (0 = perfect fairness)
    """
    unique_groups = np.unique(sensitive_attr)
    positive_rates = []
    
    for group in unique_groups:
        group_mask = sensitive_attr == group
        group_pred = y_pred[group_mask]
        positive_rate = np.mean(group_pred)
        positive_rates.append(positive_rate)
    
    # Return maximum difference between groups
    return max(positive_rates) - min(positive_rates)


def equalized_odds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray
) -> Dict[str, float]:
    """
    Calculate equalized odds metrics.
    
    Equalized odds requires equal true positive rates and false positive rates
    across groups.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attr: Sensitive attribute values
        
    Returns:
        Dictionary with TPR and FPR differences
    """
    unique_groups = np.unique(sensitive_attr)
    tpr_list = []
    fpr_list = []
    
    for group in unique_groups:
        group_mask = sensitive_attr == group
        group_y_true = y_true[group_mask]
        group_y_pred = y_pred[group_mask]
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(
            group_y_true, group_y_pred, labels=[0, 1]
        ).ravel()
        
        # True positive rate
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        # False positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return {
        "tpr_difference": max(tpr_list) - min(tpr_list),
        "fpr_difference": max(fpr_list) - min(fpr_list),
        "max_difference": max(
            max(tpr_list) - min(tpr_list),
            max(fpr_list) - min(fpr_list)
        )
    }


def equal_opportunity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray
) -> float:
    """
    Calculate equal opportunity difference.
    
    Equal opportunity requires equal true positive rates across groups.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attr: Sensitive attribute values
        
    Returns:
        Equal opportunity difference
    """
    unique_groups = np.unique(sensitive_attr)
    tpr_list = []
    
    for group in unique_groups:
        group_mask = sensitive_attr == group
        group_y_true = y_true[group_mask]
        group_y_pred = y_pred[group_mask]
        
        # Calculate TPR
        tn, fp, fn, tp = confusion_matrix(
            group_y_true, group_y_pred, labels=[0, 1]
        ).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tpr_list.append(tpr)
    
    return max(tpr_list) - min(tpr_list)


def predictive_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray
) -> float:
    """
    Calculate predictive parity difference.
    
    Predictive parity requires equal precision across groups.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attr: Sensitive attribute values
        
    Returns:
        Predictive parity difference
    """
    unique_groups = np.unique(sensitive_attr)
    precision_list = []
    
    for group in unique_groups:
        group_mask = sensitive_attr == group
        group_y_true = y_true[group_mask]
        group_y_pred = y_pred[group_mask]
        
        precision = precision_score(
            group_y_true, group_y_pred, zero_division=0
        )
        precision_list.append(precision)
    
    return max(precision_list) - min(precision_list)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all standard and fairness metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attr: Optional sensitive attribute values
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }
    
    if sensitive_attr is not None:
        metrics["demographic_parity"] = demographic_parity(y_pred, sensitive_attr)
        
        eo_metrics = equalized_odds(y_true, y_pred, sensitive_attr)
        metrics.update(eo_metrics)
        
        metrics["equal_opportunity"] = equal_opportunity(
            y_true, y_pred, sensitive_attr
        )
        metrics["predictive_parity"] = predictive_parity(
            y_true, y_pred, sensitive_attr
        )
    
    return metrics
