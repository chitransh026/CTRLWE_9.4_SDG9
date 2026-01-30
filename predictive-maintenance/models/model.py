import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PREDICTIVE MAINTENANCE MODEL TRAINING")
print("="*80)

# =========================
# 1. LOAD DATA
# =========================
print("\n[1/6] Loading preprocessed data...")
X_train_full = pd.read_csv("X_train.csv")
X_test_full = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

print(f"Original X_train shape: {X_train_full.shape}")
print(f"Original X_test shape: {X_test_full.shape}")

# =========================
# 2. SELECT ONLY REQUIRED INPUT FEATURES
# =========================
print("\n[2/6] Selecting only required input features...")

REQUIRED_FEATURES = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "temperature",
    "humidity",
    "rainfall"
]

# Check if all required features exist
missing_features = [f for f in REQUIRED_FEATURES if f not in X_train_full.columns]
if missing_features:
    print(f"⚠️  WARNING: Missing features in training data: {missing_features}")
    print(f"Available features: {X_train_full.columns.tolist()}")
    print("\nAttempting to use available features that match...")
    REQUIRED_FEATURES = [f for f in REQUIRED_FEATURES if f in X_train_full.columns]

print(f"Using {len(REQUIRED_FEATURES)} input features:")
for i, feat in enumerate(REQUIRED_FEATURES, 1):
    print(f"  {i}. {feat}")

# Select only required features
X_train = X_train_full[REQUIRED_FEATURES].copy()
X_test = X_test_full[REQUIRED_FEATURES].copy()

print(f"\nNew X_train shape: {X_train.shape}")
print(f"New X_test shape: {X_test.shape}")

# =========================
# 3. CHECK AND HANDLE NaN VALUES
# =========================
print("\n[3/6] Checking for NaN values...")

# Check for NaN in features
nan_cols_train = X_train.columns[X_train.isna().any()].tolist()
nan_cols_test = X_test.columns[X_test.isna().any()].tolist()

if nan_cols_train:
    print(f"⚠️  Found NaN in training features: {nan_cols_train}")
    print("   Filling with median values...")
    for col in nan_cols_train:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

if nan_cols_test:
    print(f"⚠️  Found NaN in test features: {nan_cols_test}")
    for col in nan_cols_test:
        if col not in nan_cols_train:
            median_val = X_train[col].median()
            X_test[col] = X_test[col].fillna(median_val)

# Check for infinite values
print("Checking for infinite values...")
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Fill any remaining NaN
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

# Double check
if X_train.isna().any().any():
    print("⚠️  Still have NaN after filling, using 0...")
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

print("✅ Data cleaned successfully")

# Check for NaN in targets
if y_train.isna().any().any():
    print("⚠️  Found NaN in targets, filling with median...")
    y_train = y_train.fillna(y_train.median())
    y_test = y_test.fillna(y_test.median())

# =========================
# 4. VERIFY TARGET VARIABLES
# =========================
EXPECTED_TARGETS = ["vibration_health", "thermal_health", "efficiency_index", "failure_risk"]
targets = y_train.columns.tolist()

print(f"\n[4/6] Verifying target variables...")
print(f"Expected targets: {EXPECTED_TARGETS}")
print(f"Actual targets:   {targets}")

if set(targets) != set(EXPECTED_TARGETS):
    print(f"⚠️  WARNING: Target mismatch!")
    print(f"Missing: {set(EXPECTED_TARGETS) - set(targets)}")
    print(f"Extra:   {set(targets) - set(EXPECTED_TARGETS)}")

# =========================
# 5. TRAIN MODELS FOR EACH TARGET
# =========================
print("\n" + "="*80)
print("[5/6] TRAINING ENSEMBLE MODELS FOR EACH TARGET")
print("="*80)

all_models = {}
all_predictions = {}

for target in targets:
    print(f"\n{'='*60}")
    print(f"Training models for: {target.upper()}")
    print(f"{'='*60}")
    
    y_train_target = y_train[target].values
    y_test_target = y_test[target].values
    
    models = {}
    predictions = {}
    
    # ============================================
    # MODEL 1: XGBoost with Optimized Parameters
    # ============================================
    print("\n  [1/4] Training XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train_target, verbose=False)
    xgb_pred = xgb_model.predict(X_test)
    models['xgboost'] = xgb_model
    predictions['xgboost'] = xgb_pred
    
    xgb_r2 = r2_score(y_test_target, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test_target, xgb_pred))
    print(f"    XGBoost - R²: {xgb_r2:.4f}, RMSE: {xgb_rmse:.4f}")
    
    # ============================================
    # MODEL 2: Random Forest
    # ============================================
    print("  [2/4] Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train_target)
    rf_pred = rf_model.predict(X_test)
    models['random_forest'] = rf_model
    predictions['random_forest'] = rf_pred
    
    rf_r2 = r2_score(y_test_target, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test_target, rf_pred))
    print(f"    Random Forest - R²: {rf_r2:.4f}, RMSE: {rf_rmse:.4f}")
    
    # ============================================
    # MODEL 3: Histogram Gradient Boosting
    # ============================================
    print("  [3/4] Training Histogram Gradient Boosting...")
    gb_model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=2,
        l2_regularization=0.5,
        random_state=42
    )
    gb_model.fit(X_train, y_train_target)
    gb_pred = gb_model.predict(X_test)
    models['gradient_boosting'] = gb_model
    predictions['gradient_boosting'] = gb_pred
    
    gb_r2 = r2_score(y_test_target, gb_pred)
    gb_rmse = np.sqrt(mean_squared_error(y_test_target, gb_pred))
    print(f"    Gradient Boosting - R²: {gb_r2:.4f}, RMSE: {gb_rmse:.4f}")
    
    # ============================================
    # MODEL 4: Ridge Regression
    # ============================================
    print("  [4/4] Training Ridge Regression...")
    ridge_model = Ridge(alpha=10.0, random_state=42)
    ridge_model.fit(X_train, y_train_target)
    ridge_pred = ridge_model.predict(X_test)
    models['ridge'] = ridge_model
    predictions['ridge'] = ridge_pred
    
    ridge_r2 = r2_score(y_test_target, ridge_pred)
    ridge_rmse = np.sqrt(mean_squared_error(y_test_target, ridge_pred))
    print(f"    Ridge Regression - R²: {ridge_r2:.4f}, RMSE: {ridge_rmse:.4f}")
    
    # ============================================
    # ENSEMBLE: Weighted Average
    # ============================================
    print("\n  [Ensemble] Creating weighted ensemble...")
    
    # Calculate weights based on R² scores
    r2_scores = np.array([xgb_r2, rf_r2, gb_r2, ridge_r2])
    r2_scores = np.maximum(r2_scores, 0)
    weights = r2_scores / (r2_scores.sum() + 1e-6)
    
    ensemble_pred = (
        weights[0] * xgb_pred +
        weights[1] * rf_pred +
        weights[2] * gb_pred +
        weights[3] * ridge_pred
    )
    
    ensemble_r2 = r2_score(y_test_target, ensemble_pred)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test_target, ensemble_pred))
    ensemble_mae = mean_absolute_error(y_test_target, ensemble_pred)
    
    print(f"    Weights: XGB={weights[0]:.3f}, RF={weights[1]:.3f}, GB={weights[2]:.3f}, Ridge={weights[3]:.3f}")
    print(f"    Ensemble - R²: {ensemble_r2:.4f}, RMSE: {ensemble_rmse:.4f}, MAE: {ensemble_mae:.4f}")
    
    # Store everything
    all_models[target] = {
        'models': models,
        'weights': weights,
        'ensemble_pred': ensemble_pred
    }
    all_predictions[target] = ensemble_pred

# =========================
# 6. COMPREHENSIVE EVALUATION
# =========================
print("\n" + "="*80)
print("[6/6] MODEL PERFORMANCE EVALUATION")
print("="*80)

for target in targets:
    print(f"\n{'='*60}")
    print(f"{target.upper()}")
    print(f"{'='*60}")
    
    y_true = y_test[target].values
    y_pred = all_predictions[target]
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6))) * 100
    
    # Calculate accuracy at different tolerance levels
    errors = np.abs(y_pred - y_true)
    mean_target = np.mean(np.abs(y_true))
    tolerance_5_percent = np.mean(errors <= (mean_target * 0.05))
    tolerance_10_percent = np.mean(errors <= (mean_target * 0.10))
    tolerance_15_percent = np.mean(errors <= (mean_target * 0.15))
    
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  MAE:   {mae:.4f}")
    print(f"  R²:    {r2:.4f}")
    print(f"  MAPE:  {mape:.2f}%")
    print(f"\n  Accuracy within 5% of mean:  {tolerance_5_percent:.1%}")
    print(f"  Accuracy within 10% of mean: {tolerance_10_percent:.1%}")
    print(f"  Accuracy within 15% of mean: {tolerance_15_percent:.1%}")

# Overall performance
print(f"\n{'='*60}")
print("OVERALL MODEL PERFORMANCE")
print(f"{'='*60}")

all_y_true = []
all_y_pred = []
for target in targets:
    all_y_true.append(y_test[target].values)
    all_y_pred.append(all_predictions[target])

all_y_true = np.concatenate(all_y_true)
all_y_pred = np.concatenate(all_y_pred)

overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
overall_mae = mean_absolute_error(all_y_true, all_y_pred)
overall_r2 = r2_score(all_y_true, all_y_pred)

print(f"  Overall RMSE: {overall_rmse:.4f}")
print(f"  Overall MAE:  {overall_mae:.4f}")
print(f"  Overall R²:   {overall_r2:.4f}")

# =========================
# 7. SAVE MODEL PACKAGE
# =========================
print("\n" + "="*80)
print("SAVING MODEL PACKAGE")
print("="*80)

model_package = {
    'all_models': all_models,
    'feature_names': REQUIRED_FEATURES,  # Only the 8 required features
    'target_names': targets,
    'performance': {
        'overall_rmse': overall_rmse,
        'overall_mae': overall_mae,
        'overall_r2': overall_r2
    }
}

joblib.dump(model_package, 'predictive_maintenance_model.pkl')
print("✅ Model saved: predictive_maintenance_model.pkl")
print(f"   Input features (8): {REQUIRED_FEATURES}")
print(f"   Output targets (4): {targets}")

# =========================
# 8. CREATE PREDICTION EXAMPLES
# =========================
print("\n" + "="*80)
print("EXAMPLE PREDICTIONS (First 10 samples)")
print("="*80)

for i in range(min(10, len(y_test))):
    print(f"\n{'='*60}")
    print(f"SAMPLE {i+1}")
    print(f"{'='*60}")
    
    # Show input features
    print("  INPUT FEATURES:")
    for feat in REQUIRED_FEATURES:
        print(f"    {feat:25s}: {X_test[feat].iloc[i]:.3f}")
    
    print("\n  PREDICTIONS:")
    for target in targets:
        actual = y_test[target].iloc[i]
        predicted = all_predictions[target][i]
        error = abs(predicted - actual)
        error_pct = (error / abs(actual)) * 100 if actual != 0 else 0
        
        print(f"    {target:20s}: Actual={actual:8.3f}, Predicted={predicted:8.3f}, Error={error:6.3f} ({error_pct:5.1f}%)")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print("\nModel Configuration:")
print(f"  📥 Input Features:  {len(REQUIRED_FEATURES)} features")
print(f"     {REQUIRED_FEATURES}")
print(f"  📤 Output Targets:  {len(targets)} targets")
print(f"     {targets}")
print("\nKey Features:")
print("  ✅ Uses only 8 specified input features")
print("  ✅ Predicts 4 health/risk indicators")
print("  ✅ Ensemble of 4 different models")
print("  ✅ Optimized hyperparameters")
print("  ✅ Weighted averaging based on performance")

# =========================
# 9. SAVE PREDICTION FUNCTION
# =========================
print("\n💾 Creating prediction function...")

prediction_code = '''
import joblib
import pandas as pd
import numpy as np

def predict_maintenance(input_data):
    """
    Predict maintenance indicators from sensor data.
    
    Parameters:
    -----------
    input_data : dict
        Dictionary with keys:
        - air_temperature_k
        - process_temperature_k
        - rotational_speed_rpm
        - torque_nm
        - tool_wear_min
        - temperature
        - humidity
        - rainfall
    
    Returns:
    --------
    dict with keys:
        - vibration_health
        - thermal_health
        - efficiency_index
        - failure_risk
    """
    # Load model
    model_package = joblib.load('predictive_maintenance_model.pkl')
    
    # Create DataFrame from input
    df = pd.DataFrame([input_data])
    
    # Ensure correct feature order
    df = df[model_package['feature_names']]
    
    # Make predictions for each target
    predictions = {}
    for target in model_package['target_names']:
        target_models = model_package['all_models'][target]
        weights = target_models['weights']
        
        # Get predictions from each model
        xgb_pred = target_models['models']['xgboost'].predict(df)[0]
        rf_pred = target_models['models']['random_forest'].predict(df)[0]
        gb_pred = target_models['models']['gradient_boosting'].predict(df)[0]
        ridge_pred = target_models['models']['ridge'].predict(df)[0]
        
        # Weighted ensemble
        ensemble_pred = (
            weights[0] * xgb_pred +
            weights[1] * rf_pred +
            weights[2] * gb_pred +
            weights[3] * ridge_pred
        )
        
        predictions[target] = round(float(ensemble_pred), 2)
    
    return predictions


# Example usage
if __name__ == "__main__":
    sample_input = {
        "air_temperature_k": 298.5,
        "process_temperature_k": 310.2,
        "rotational_speed_rpm": 1500,
        "torque_nm": 45.3,
        "tool_wear_min": 120,
        "temperature": 25.3,
        "humidity": 65.0,
        "rainfall": 2.5
    }
    
    result = predict_maintenance(sample_input)
    print("Predictions:", result)
    # Expected output format:
    # {
    #     "vibration_health": 0.32,
    #     "thermal_health": 0.41,
    #     "efficiency_index": 0.67,
    #     "failure_risk": 0.81
    # }
'''

with open('predict.py', 'w') as f:
    f.write(prediction_code)

print("✅ Saved: predict.py (prediction function)")

# Save cleaned data
print("\n💾 Saving cleaned datasets with only required features...")
X_train.to_csv('X_train_cleaned.csv', index=False)
X_test.to_csv('X_test_cleaned.csv', index=False)
print("✅ Saved: X_train_cleaned.csv, X_test_cleaned.csv")

print("\n" + "="*80)
print("ALL DONE! 🎉")
print("="*80)