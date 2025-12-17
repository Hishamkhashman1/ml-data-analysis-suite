import time
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def train_linear_regression(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    manual: bool = False,
    learning_rate: float = 0.01,
    epochs: int = 100,
    batch_size: int = 32,
) -> Tuple[LinearRegression, Dict[str, float], Dict[str, float]]:
    """Train a linear regression model and return the model and metrics."""
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    training_time = time.time() - start_time

    if manual:
        print(f"Supervised Training - MSE: {mse}, MAE: {mae}")
    else:
        print(f"Automatic Training - MSE: {mse}, MAE: {mae}")

    loss_table = {"MSE": mse, "MAE": mae}
    training_stats = {
        "Time to execute": training_time,
        "Batch size used": batch_size,
        "Epochs used": epochs,
        "Learning rate used": learning_rate,
        "MSE": mse,
        "MAE": mae,
        "Prediction formula": f"y = {model.intercept_} + {model.coef_} * X",
    }

    return model, loss_table, training_stats
