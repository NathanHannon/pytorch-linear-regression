import torch
import torch.nn as nn
from ..utils.data_loader import load_kc_house_data

# import matplotlib.pyplot as plt # No longer needed here
import os
import logging


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):  # Added hidden_dim
        super(LinearRegressionModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)  # New hidden layer
        self.relu = nn.ReLU()  # Activation function
        self.linear2 = nn.Linear(hidden_dim, output_dim)  # Output layer

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out


def train_save_get_data(data_path, model_save_path, params_save_path):
    """
    Loads data, trains a linear regression model, saves the model state
    and normalization parameters, yielding progress updates and finally
    returning plot data.

    Args:
        data_path (str): Path to the kc_house_data.csv file.
        model_save_path (str): Path to save the trained model state dictionary.
        params_save_path (str): Path to save normalization parameters.

    Yields:
        dict: Progress updates {'type': 'progress', 'epoch': ..., 'epochs': ..., 'loss': ...}
              or final result {'type': 'result', 'success': ..., ...}
    """
    try:
        logging.info(f"Loading data from {data_path}")
        yield {"type": "progress", "message": "Loading data..."}
        df = load_kc_house_data(data_path)
        # ... (keep data loading and normalization as is) ...
        df_clean = df[["sqft_living", "price"]].dropna()
        X = df_clean[["sqft_living"]].values
        y = df_clean[["price"]].values

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        logging.info("Normalizing data...")
        yield {"type": "progress", "message": "Normalizing data..."}
        X_mean, X_std = X_tensor.mean(), X_tensor.std()
        X_tensor_normalized = (X_tensor - X_mean) / (X_std + 1e-8)

        y_mean, y_std = y_tensor.mean(), y_tensor.std()
        y_tensor_normalized = (y_tensor - y_mean) / (y_std + 1e-8)

        logging.info("Initializing model, loss, and optimizer...")
        yield {"type": "progress", "message": "Initializing model..."}
        input_dim = 1
        hidden_dim = 10  # New hidden dimension
        output_dim = 1
        # hidden_dim is defined above (e.g., hidden_dim = 10)
        model = LinearRegressionModel(
            input_dim, hidden_dim, output_dim
        )  # Pass hidden_dim here
        learning_rate = 0.01
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        logging.info("Starting training loop...")
        epochs = 2500
        yield {
            "type": "progress",
            "message": "Starting training...",
            "epoch": 0,
            "epochs": epochs,
        }  # Initial message
        for epoch in range(epochs):
            outputs = model(X_tensor_normalized)
            loss = criterion(outputs, y_tensor_normalized)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Yield progress update periodically (e.g., every 20 epochs)
            if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
                progress_update = {
                    "type": "progress",
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "loss": f"{loss.item():.4f}",  # Format loss
                }
                yield progress_update
                # Optional: Add a small delay to allow frontend to update smoothly
                # time.sleep(0.01)

        logging.info("Training finished.")
        yield {"type": "progress", "message": "Saving model..."}  # Saving message

        # --- Saving Model and Parameters ---
        # ... (keep saving logic as is) ...
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

        yield {
            "type": "progress",
            "message": "Preparing plot data...",
        }  # Plotting message

        # --- Preparing Plot Data ---
        # ... (keep plot data preparation as is) ...
        logging.info("Preparing plot data...")
        model.eval()
        with torch.no_grad():
            predicted_normalized = model(X_tensor_normalized)
        predicted = (predicted_normalized * y_std) + y_mean
        original_data = [{"x": float(X[i]), "y": float(y[i])} for i in range(len(X))]
        line_data_unsorted = [
            {"x": float(X[i]), "y": float(predicted[i])} for i in range(len(X))
        ]
        fitted_line_data = sorted(line_data_unsorted, key=lambda p: p["x"])

        # Yield the final result
        yield {
            "type": "result",  # Mark as final result
            "success": True,
            "original_data": original_data,
            "fitted_line": fitted_line_data,
        }

    except FileNotFoundError:
        logging.error(f"Error: Could not find the data file at {data_path}")
        yield {"type": "result", "success": False, "message": "Data file not found."}
    except Exception as e:
        logging.error(f"An error occurred during training: {e}", exc_info=True)
        yield {"type": "result", "success": False, "message": f"Training error: {e}"}


# Remove the if __name__ == "__main__": block or comment it out
