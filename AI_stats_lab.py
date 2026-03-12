"""
AIstats_lab.py

Student starter file for the Regularization & Overfitting lab.
"""

# =========================
# Libraries
# =========================

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# =========================
# Helper Functions
# =========================

def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# =========================
# Q1 Lasso Regression
# =========================

def lasso_regression_diabetes(lambda_reg=0.1, lr=0.01, epochs=2000):
    """
    Implement Lasso regression using gradient descent.
    """

    # Load diabetes dataset
    dataset = load_diabetes()
    X = dataset.data
    y = dataset.target

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Add bias column
    X_train = add_bias(X_train)
    X_test = add_bias(X_test)

    # Initialize theta
    n_samples, n_features = X_train.shape
    theta = np.zeros(n_features)

    # Gradient Descent with L1 Regularization
    for _ in range(epochs):

        predictions = X_train @ theta
        errors = predictions - y_train

        # Gradient of MSE
        gradient = (X_train.T @ errors) / n_samples

        # L1 regularization term (subgradient)
        l1_penalty = lambda_reg * np.sign(theta)

        # Do not regularize bias
        l1_penalty[0] = 0

        # Parameter update
        theta = theta - lr * (gradient + l1_penalty)

    # Predictions
    train_predictions = X_train @ theta
    test_predictions = X_test @ theta

    # Metrics
    train_mse = mse(y_train, train_predictions)
    test_mse = mse(y_test, test_predictions)

    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)

    return train_mse, test_mse, train_r2, test_r2, theta


# =========================
# Q2 Polynomial Overfitting
# =========================

def polynomial_overfitting_experiment(max_degree=10):
    """
    Study overfitting using polynomial regression.
    """

    # Load dataset
    dataset = load_diabetes()

    # Use BMI feature only (index = 2)
    X = dataset.data[:, 2].reshape(-1, 1)
    y = dataset.target

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    degrees = []
    train_errors = []
    test_errors = []

    # Loop through polynomial degrees
    for degree in range(1, max_degree + 1):

        degrees.append(degree)

        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=True)

        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # Normal Equation for Linear Regression
        theta = np.linalg.pinv(X_train_poly.T @ X_train_poly) @ X_train_poly.T @ y_train

        # Predictions
        train_predictions = X_train_poly @ theta
        test_predictions = X_test_poly @ theta

        # Compute errors
        train_mse = mse(y_train, train_predictions)
        test_mse = mse(y_test, test_predictions)

        train_errors.append(train_mse)
        test_errors.append(test_mse)

    return {
        "degrees": degrees,
        "train_mse": train_errors,
        "test_mse": test_errors
    }
