import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_rmse(y_true, y_pred):
    """Root Mean Squared Error between actual and predicted values."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def directional_accuracy(y_true, y_pred):
    """% of times the model correctly predicted up/down direction."""
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    return np.mean(true_direction == pred_direction) * 100