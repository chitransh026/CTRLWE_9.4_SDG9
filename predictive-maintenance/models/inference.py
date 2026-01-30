import joblib
import pandas as pd
import numpy as np
import os

# ==============================
# Utility: Normalize to 0–100
# ==============================
def normalize_to_0_100(value, lower_bound, upper_bound):
    """
    Robust normalization to 0–100 with clipping.
    """
    if upper_bound == lower_bound:
        return 50.0

    norm = (value - lower_bound) / (upper_bound - lower_bound)
    norm = max(0.0, min(1.0, norm))
    return round(norm * 100, 2)

# ==============================
# Load Model Package
# ==============================
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "predictive_maintenance_model.pkl"
)

def load_model():
    return joblib.load(MODEL_PATH)

model_package = load_model()

REQUIRED_FEATURES = model_package["feature_names"]
TARGETS = model_package["target_names"]

# ==============================
# Normalization Bounds (Dynamic)
# ==============================
# Create bounds dynamically based on actual target names
BOUNDS = {}
for target in TARGETS:
    if "vibration" in target.lower():
        BOUNDS[target] = (-2000, 2000)
    elif "thermal" in target.lower():
        BOUNDS[target] = (-1500, 1500)
    elif "efficiency" in target.lower():
        BOUNDS[target] = (-2000, 2000)
    elif "failure" in target.lower() or "risk" in target.lower():
        BOUNDS[target] = (-2000, 2000)
    else:
        # Default bounds for unknown targets
        BOUNDS[target] = (-2000, 2000)

# ==============================
# Main Prediction Function
# ==============================
def predict_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs ensemble prediction for all rows in dataframe.
    Returns dataframe with 4 prediction columns (0–100 scale).
    """

    # Check for missing required features
    missing_features = set(REQUIRED_FEATURES) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}. "
                        f"Required features are: {REQUIRED_FEATURES}")

    # Ensure correct feature order
    df = df[REQUIRED_FEATURES].copy()

    # Handle NaN / infinite values safely
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median()).fillna(0)

    results = {target: [] for target in TARGETS}

    for _, row in df.iterrows():
        row_df = pd.DataFrame([row])

        for target in TARGETS:
            target_block = model_package["all_models"][target]
            models = target_block["models"]
            weights = target_block["weights"]

            preds = np.array([
                models["xgboost"].predict(row_df)[0],
                models["random_forest"].predict(row_df)[0],
                models["gradient_boosting"].predict(row_df)[0],
                models["ridge"].predict(row_df)[0]
            ])

            ensemble_pred = np.dot(weights, preds)

            # Normalize to 0–100
            low, high = BOUNDS[target]
            score_0_100 = normalize_to_0_100(ensemble_pred, low, high)

            results[target].append(score_0_100)

    return pd.DataFrame(results)

# ==============================
# CLI Test
# ==============================
if __name__ == "__main__":
    test_df = pd.DataFrame([{
        "air_temperature_k": 298.5,
        "process_temperature_k": 310.2,
        "rotational_speed_rpm": 1500,
        "torque_nm": 45.3,
        "tool_wear_min": 120,
        "temperature": 25.3,
        "humidity": 65,
        "rainfall": 2.5
    }])

    print("\n🔍 Test Prediction Output:")
    print(predict_from_dataframe(test_df))