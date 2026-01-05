"""Tests for fairness metrics."""

import numpy as np
import pytest
from src.fairness import (
    demographic_parity,
    equalized_odds,
    equal_opportunity,
    predictive_parity,
    compute_all_metrics,
)


def test_demographic_parity_perfect_fairness():
    """Test demographic parity with perfect fairness."""
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    sensitive_attr = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    
    dp = demographic_parity(y_pred, sensitive_attr)
    assert dp == 0.0, "Perfect fairness should have DP = 0"


def test_demographic_parity_unfair():
    """Test demographic parity with unfair predictions."""
    y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    sensitive_attr = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    dp = demographic_parity(y_pred, sensitive_attr)
    assert dp == 1.0, "Complete disparity should have DP = 1"


def test_equalized_odds():
    """Test equalized odds calculation."""
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    sensitive_attr = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    
    eo = equalized_odds(y_true, y_pred, sensitive_attr)
    assert "tpr_difference" in eo
    assert "fpr_difference" in eo
    assert "max_difference" in eo


def test_equal_opportunity():
    """Test equal opportunity calculation."""
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    sensitive_attr = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    
    eo = equal_opportunity(y_true, y_pred, sensitive_attr)
    assert eo == 0.0, "Perfect predictions should have EO = 0"


def test_compute_all_metrics():
    """Test computing all metrics."""
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    sensitive_attr = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    
    metrics = compute_all_metrics(y_true, y_pred, sensitive_attr)
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "demographic_parity" in metrics
    assert "equal_opportunity" in metrics
    
    assert metrics["accuracy"] == 1.0, "Perfect predictions should have accuracy = 1"


def test_compute_metrics_without_sensitive_attr():
    """Test computing metrics without sensitive attributes."""
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    
    metrics = compute_all_metrics(y_true, y_pred)
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "demographic_parity" not in metrics
