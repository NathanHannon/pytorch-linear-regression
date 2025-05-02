from flask import Flask, render_template, url_for, jsonify, request
import os
import sys

# import time # No longer needed for cache busting plot image
import logging
import torch
import torch.nn as nn

# --- Path Setup ---
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# --- Import Model Function and Definition ---
try:
    # Import the updated training function
    from src.models.linear_regression import train_save_get_data  # UPDATED import
    from src.models.linear_regression import LinearRegressionModel
except ImportError as e:
    logging.error(
        f"Failed to import from src.models.linear_regression: {e}", exc_info=True
    )

    # Define dummy function and class if import fails
    def train_save_get_data(*args, **kwargs):  # UPDATED dummy
        logging.error("Using dummy train_save_get_data due to import failure.")
        return {"success": False, "message": "Dummy function used."}

    class LinearRegressionModel(nn.Module):  # Dummy model
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.linear = nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)


# --- Configuration ---
template_dir = os.path.join(project_root, "src", "templates")
static_dir = os.path.join(project_root, "static")  # Keep static for CSS/JS if needed
data_path = os.path.join(project_root, "data", "kc_house_data.csv")
# plot_filename = "results.png" # No longer generating static plot
model_filename = "linear_regression_model.pth"
params_filename = "normalization_params.pt"

# plot_save_path = os.path.join(static_dir, plot_filename) # No longer needed
model_save_path = os.path.join(project_root, "models", model_filename)
params_save_path = os.path.join(project_root, "models", params_filename)

# Ensure directories exist
os.makedirs(static_dir, exist_ok=True)
os.makedirs(os.path.join(project_root, "models"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Flask App Initialization ---
app = Flask(
    __name__,
    template_folder=template_dir,
    static_folder=static_dir,
    static_url_path="/static",
)


# --- Helper: Load Model and Params ---
# ... (keep load_model_and_params as is) ...
def load_model_and_params():
    """Loads the saved model state and normalization parameters."""
    if not os.path.exists(model_save_path) or not os.path.exists(params_save_path):
        return None, None

    try:
        # Load parameters first
        norm_params = torch.load(params_save_path)

        # Instantiate model and load state dict
        model = LinearRegressionModel(input_dim=1, output_dim=1)
        model.load_state_dict(torch.load(model_save_path))
        model.eval()  # Set to evaluation mode
        return model, norm_params
    except Exception as e:
        logging.error(f"Error loading model or params: {e}", exc_info=True)
        return None, None


# --- Routes ---
@app.route("/")
def index():
    """Renders the main page."""
    logging.info("Request received for index page.")
    model_exists = os.path.exists(model_save_path)
    # No need to check for plot image anymore
    return render_template("index.html", model_exists=model_exists)  # Remove plot_url


@app.route("/train", methods=["POST"])
def train_model():
    """Triggers the model training process and returns plot data."""
    logging.info("Received request to train model.")
    if not os.path.isfile(data_path):
        logging.error(f"Data file not found at {data_path}")
        return jsonify(
            {"success": False, "message": f"Server error: Data file not found."}
        ), 500

    # Call the updated function
    result_data = train_save_get_data(data_path, model_save_path, params_save_path)

    if result_data and result_data["success"]:
        logging.info("Training successful. Sending plot data.")
        # Return the whole dictionary which includes success status and data
        return jsonify(result_data)
    else:
        logging.error(
            f"Model training failed. Reason: {result_data.get('message', 'Unknown')}"
        )
        return jsonify(result_data), 500  # Send error details back


@app.route("/predict", methods=["POST"])
# ... (keep predict_price route as is) ...
def predict_price():
    """Predicts price based on input square footage."""
    logging.info("Received request to predict price.")
    model, norm_params = load_model_and_params()

    if model is None or norm_params is None:
        logging.warning("Prediction attempt failed: Model or params not loaded.")
        return jsonify(
            {"success": False, "message": "Model not trained yet or failed to load."}
        ), 400

    try:
        data = request.get_json()
        if not data or "sqft" not in data:
            logging.warning("Prediction failed: 'sqft' missing in request.")
            return jsonify(
                {"success": False, "message": 'Missing "sqft" in request data.'}
            ), 400

        sqft_input = float(data["sqft"])
        logging.info(f"Predicting for sqft: {sqft_input}")

        # Prepare input tensor
        X_input = torch.tensor([[sqft_input]], dtype=torch.float32)

        # Normalize input using loaded parameters
        X_input_normalized = (X_input - norm_params["X_mean"]) / (
            norm_params["X_std"] + 1e-8
        )

        # Predict
        with torch.no_grad():
            predicted_normalized = model(X_input_normalized)

        # Denormalize output using loaded parameters
        predicted_price = (predicted_normalized * norm_params["y_std"]) + norm_params[
            "y_mean"
        ]
        predicted_price_value = predicted_price.item()  # Get scalar value

        logging.info(f"Predicted price: {predicted_price_value:.2f}")
        return jsonify(
            {"success": True, "prediction": f"${predicted_price_value:,.2f}"}
        )  # Format as currency

    except ValueError:
        logging.warning("Prediction failed: Invalid sqft input.")
        return jsonify(
            {"success": False, "message": "Invalid square footage value provided."}
        ), 400
    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}", exc_info=True)
        return jsonify(
            {
                "success": False,
                "message": "An internal error occurred during prediction.",
            }
        ), 500


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting Flask development server...")
    app.run(debug=True, host="0.0.0.0", port=5000)
