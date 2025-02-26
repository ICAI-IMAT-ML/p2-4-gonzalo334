import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)


        X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            return self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # Replace this code with the code you did in the previous laboratory session
        #columna_intercept = np.ones((X.shape[0], 1))
        #X = np.hstack((columna_intercept, X))

        X_transpose = np.transpose(X)
        
        w = np.dot(np.dot(np.linalg.inv(np.dot(X_transpose, X)), X_transpose), y)

        self.intercept = w[0]
        self.coefficients = w[1:]


    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using gradient descent and track progress.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Updates the model's coefficients and intercept in-place.
        """
        m = len(y)  # Number of samples
        self.coefficients = np.random.rand(X.shape[1] - 1) * 0.01  # Small random numbers
        self.intercept = np.random.rand() * 0.01

        loss_history = []
        w_history = []
        b_history = []                

        for epoch in range(iterations):
            # Compute predictions
            predictions = np.dot(X, np.hstack([self.intercept, self.coefficients]))

            # Compute error
            error = predictions - y


            # Compute gradients
            gradient = (1 / m) * np.dot(np.transpose(X), error)            
            
            # Update parameters
            self.intercept -= learning_rate * gradient[0]
            self.coefficients -= learning_rate * gradient[1:]

            # Compute and store loss
            mse = 1 / m * np.sum(error ** 2)
            loss_history.append(mse)
            w_history.append(self.coefficients.copy())
            b_history.append(self.intercept)

            
            # Print progress
            if epoch % 100 == 0:
                print(f"Iteration {epoch}: MSE = {mse}")

        return loss_history, b_history, w_history

    

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            fit (bool): Flag to indicate if fit was done.

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """

        # Paste your code from last week

        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")
        
        if np.ndim(X) == 1:
            return self.intercept + self.coefficients * X
        else:
            return self.intercept + np.dot(X, self.coefficients)

        


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """

    # R^2 Score
    rss = np.sum((y_true-y_pred)**2)
    tss = np.sum((y_true-y_true.mean())**2)
    r_squared = 1 - rss / tss 
    
    # Root Mean Squared Error
    mse = 1/y_true.shape[0] * np.sum((y_true-y_pred)**2)
    rmse = np.sqrt(mse)

    # Mean Absolute Error
    mae = 1/y_true.shape[0] * np.sum(abs(y_true-y_pred))  #   Podríamos usar len(y_true) o la media en vez de 1/N y sum

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    X_transformed = X.copy()
    for index in sorted(categorical_indices, reverse=True):
        # Extract the categorical column
        categorical_column = X_transformed[:, index]

        # Find the unique categories (works with strings)
        unique_values = np.unique(categorical_column)
        
        # Optionally drop the first level of one-hot encoding
        if drop_first:
            unique_values = unique_values[1:]

        # Create a one-hot encoded matrix (np.array) for the current categorical column
        one_hot = (categorical_column[:,None]== unique_values).astype(int)

        
        # Delete the original categorical column from X_transformed and insert new one-hot encoded columns
        X_transformed = np.delete(X_transformed, index, axis=1)
        X_transformed = np.hstack((X_transformed[:,:index], one_hot, X_transformed[:, index:]))
    return X_transformed