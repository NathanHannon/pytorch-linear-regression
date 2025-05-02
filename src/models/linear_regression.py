import torch
import torch.nn as nn
from ..utils.data_loader import load_kc_house_data

# import matplotlib.pyplot as plt # No longer needed here
import os
import logging


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


# Function to perform training, saving model/params, and returning plot data
def train_save_get_data(data_path, model_save_path, params_save_path):
    """
    Loads data, trains a linear regression model, saves the model state
    and normalization parameters, and returns data for plotting.

    Args:
        data_path (str): Path to the kc_house_data.csv file.
        model_save_path (str): Path to save the trained model state dictionary.
        params_save_path (str): Path to save normalization parameters.

    Returns:
        dict: A dictionary containing plot data {'original': [...], 'predicted': [...]}
              and success status {'success': True/False}, or None on failure.
    """
    try:
        logging.info(f"Loading data from {data_path}")
        df = load_kc_house_data(data_path)
        df_clean = df[["sqft_living", "price"]].dropna()
        X = df_clean[["sqft_living"]].values
        y = df_clean[["price"]].values

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        logging.info("Normalizing data...")
        X_mean, X_std = X_tensor.mean(), X_tensor.std()
        X_tensor_normalized = (X_tensor - X_mean) / (X_std + 1e-8)

        y_mean, y_std = y_tensor.mean(), y_tensor.std()
        y_tensor_normalized = (y_tensor - y_mean) / (y_std + 1e-8)

        logging.info("Initializing model, loss, and optimizer...")
        input_dim = 1
        output_dim = 1
        model = LinearRegressionModel(input_dim, output_dim)
        learning_rate = 0.01
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        logging.info("Starting training loop...")
        epochs = 1000
        for epoch in range(epochs):
            outputs = model(X_tensor_normalized)
            loss = criterion(outputs, y_tensor_normalized)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 200 == 0:
                logging.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
        logging.info("Training finished.")

        # --- Saving Model and Parameters ---
        logging.info(f"Saving model state to {model_save_path}")
        torch.save(model.state_dict(), model_save_path)

        logging.info(f"Saving normalization parameters to {params_save_path}")
        norm_params = {
            "X_mean": X_mean,
            "X_std": X_std,
            "y_mean": y_mean,
            "y_std": y_std,
        }
        torch.save(norm_params, params_save_path)
        # ------------------------------------

        logging.info("Preparing plot data...")
        model.eval()
        with torch.no_grad():
            predicted_normalized = model(X_tensor_normalized)
        predicted = (predicted_normalized * y_std) + y_mean

        # Prepare data for JSON serialization (list of {x, y} objects)
        original_data = [{"x": float(X[i]), "y": float(y[i])} for i in range(len(X))]
        # For the line, we need pairs of (x, predicted_y)
        # Sort by X for a clean line plot
        line_data_unsorted = [
            {"x": float(X[i]), "y": float(predicted[i])} for i in range(len(X))
        ]
        fitted_line_data = sorted(line_data_unsorted, key=lambda p: p["x"])

        return {
            "success": True,
            "original_data": original_data,
            "fitted_line": fitted_line_data,
        }

    except FileNotFoundError:
        logging.error(f"Error: Could not find the data file at {data_path}")
        return {"success": False, "message": "Data file not found."}
    except Exception as e:
        logging.error(f"An error occurred during training: {e}", exc_info=True)
        return {"success": False, "message": f"Training error: {e}"}


# Remove the if __name__ == "__main__": block or comment it out
