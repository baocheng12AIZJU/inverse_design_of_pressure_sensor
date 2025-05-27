import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


def mean_relative_error(y_true, y_pred, epsilon=np.finfo(float).eps):
    """
    Compute the mean relative error (MRE).
    """
    relative_errors = np.abs((y_true - y_pred) / (y_true + epsilon))
    return np.mean(relative_errors)


def evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    Evaluate a model's performance using MRE for training and validation sets.
    """
    mre_train = mean_relative_error(y_train, model.predict(X_train))
    mre_val = mean_relative_error(y_val, model.predict(X_val))
    return mre_train, mre_val


def create_and_train_model(model_type, X_train, y_train):
    """
    Create and train a regression model based on the specified type.
    """
    if model_type == 'decision_tree':
        model = DecisionTreeRegressor(random_state=42, max_depth=5)
    elif model_type == 'gradient_boosting':
        model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
    else:
        raise ValueError("Invalid model type")

    model.fit(X_train, y_train)
    return model
