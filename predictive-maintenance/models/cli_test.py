import pandas as pd
import numpy as np
import joblib
import sys

print("="*80)
print("PREDICTIVE MAINTENANCE - CLI PREDICTION TOOL")
print("="*80)

# =========================
# 1. LOAD MODEL PACKAGE
# =========================
print("\n[1/4] Loading model...")
try:
    model_package = joblib.load('predictive_maintenance_model.pkl')
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("❌ Error: predictive_maintenance_model.pkl not found!")
    print("   Make sure you ran train_model_fixed.py first")
    sys.exit(1)

all_models = model_package['all_models']
feature_names = model_package['feature_names']
target_names = model_package['target_names']

print(f"   Targets: {target_names}")
print(f"   Features: {len(feature_names)}")

# =========================
# 2. CHECK COMMAND LINE INPUT
# =========================
print("\n[2/4] Processing input...")

if len(sys.argv) < 2:
    print("\n" + "="*80)
    print("USAGE OPTIONS")
    print("="*80)
    print(f"\nOption 1: Provide CSV file")
    print(f"  python {sys.argv[0]} input_data.csv")
    print(f"\nOption 2: Provide feature values")
    print(f"  python {sys.argv[0]} <value1> <value2> ... <value{len(feature_names)}>")
    print(f"\nRequired {len(feature_names)} features:")
    for i, feat in enumerate(feature_names, 1):
        print(f"  {i:2d}. {feat}")
    sys.exit(1)

# Check if input is a CSV file
input_arg = sys.argv[1]

if input_arg.endswith('.csv'):
    # =========================
    # OPTION 1: CSV FILE INPUT
    # =========================
    print(f"   Input type: CSV file")
    print(f"   File: {input_arg}")
    
    try:
        input_df = pd.read_csv(input_arg)
        print(f"✅ Loaded {len(input_df)} samples from CSV")
    except FileNotFoundError:
        print(f"❌ Error: File '{input_arg}' not found!")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        sys.exit(1)
    
    # Verify columns match
    if list(input_df.columns) != feature_names:
        print("\n⚠️  Warning: Column names don't match exactly")
        print("   Attempting to reorder columns...")
        
        missing_features = set(feature_names) - set(input_df.columns)
        extra_features = set(input_df.columns) - set(feature_names)
        
        if missing_features:
            print(f"   Missing features: {missing_features}")
            for feat in missing_features:
                input_df[feat] = 0
        
        if extra_features:
            print(f"   Extra features (ignored): {extra_features}")
        
        input_df = input_df[feature_names]
        print("✅ Columns reordered")

else:
    # =========================
    # OPTION 2: COMMAND LINE VALUES
    # =========================
    print(f"   Input type: Command line values")
    
    try:
        input_values = [float(x) for x in sys.argv[1:]]
    except ValueError:
        print("❌ Error: All inputs must be numeric values")
        sys.exit(1)
    
    if len(input_values) != len(feature_names):
        print(f"❌ Error: Expected {len(feature_names)} features, got {len(input_values)}")
        print(f"\nRequired features ({len(feature_names)}):")
        for i, feat in enumerate(feature_names, 1):
            print(f"  {i:2d}. {feat}")
        sys.exit(1)
    
    input_df = pd.DataFrame([input_values], columns=feature_names)
    print(f"✅ Parsed {len(input_values)} feature values")

# =========================
# 3. MAKE PREDICTIONS
# =========================
print("\n[3/4] Making predictions...")

predictions_dict = {}

for target in target_names:
    models = all_models[target]['models']
    weights = all_models[target]['weights']
    
    # Get predictions from each model
    xgb_pred = models['xgboost'].predict(input_df)
    rf_pred = models['random_forest'].predict(input_df)
    gb_pred = models['gradient_boosting'].predict(input_df)
    ridge_pred = models['ridge'].predict(input_df)
    
    # Ensemble prediction
    ensemble_pred = (
        weights[0] * xgb_pred +
        weights[1] * rf_pred +
        weights[2] * gb_pred +
        weights[3] * ridge_pred
    )
    
    predictions_dict[target] = ensemble_pred

print("✅ Predictions complete")

# =========================
# 4. DISPLAY RESULTS
# =========================
print("\n" + "="*80)
print("[4/4] PREDICTION RESULTS")
print("="*80)

# Create results DataFrame
results_df = pd.DataFrame(predictions_dict)

# Display each sample
for i in range(len(results_df)):
    print(f"\n{'='*60}")
    print(f"SAMPLE {i+1}")
    print(f"{'='*60}")
    
    # Show input features (only if single sample or command line input)
    if len(results_df) == 1 or not input_arg.endswith('.csv'):
        print("\nInput Features:")
        for feat, val in zip(feature_names, input_df.iloc[i]):
            print(f"  {feat:30s}: {val:10.4f}")
    
    # Show predictions
    print(f"\n{'PREDICTIONS':^60}")
    print("-" * 60)
    for target in target_names:
        value = results_df[target].iloc[i]
        print(f"  {target:30s}: {value:10.3f}")
    
    # Interpretation
    print(f"\n{'INTERPRETATION':^60}")
    print("-" * 60)
    
    vib = results_df['vibration_index'].iloc[i]
    therm = results_df['thermal_index'].iloc[i]
    eff = results_df['efficiency_index'].iloc[i]
    
    # Vibration health
    if vib < 5:
        vib_status = "✅ GOOD - Low vibration"
    elif vib < 15:
        vib_status = "⚠️  MODERATE - Monitor closely"
    else:
        vib_status = "❌ HIGH - Maintenance recommended"
    
    # Thermal health
    if therm < 5:
        therm_status = "✅ GOOD - Normal temperature"
    elif therm < 15:
        therm_status = "⚠️  MODERATE - Check cooling"
    else:
        therm_status = "❌ HIGH - Thermal stress detected"
    
    # Efficiency
    if eff < 5:
        eff_status = "✅ EXCELLENT - Optimal efficiency"
    elif eff < 15:
        eff_status = "⚠️  MODERATE - Slight degradation"
    else:
        eff_status = "❌ LOW - Significant degradation"
    
    print(f"  Vibration:  {vib_status}")
    print(f"  Thermal:    {therm_status}")
    print(f"  Efficiency: {eff_status}")
    
    # Overall recommendation
    print(f"\n{'RECOMMENDATION':^60}")
    print("-" * 60)
    
    max_value = max(vib, therm, eff)
    if max_value < 5:
        print("  ✅ Equipment is in excellent condition")
        print("  → Continue normal operation")
    elif max_value < 15:
        print("  ⚠️  Equipment showing moderate wear")
        print("  → Schedule inspection within 1-2 weeks")
    else:
        print("  ❌ Equipment requires attention")
        print("  → Schedule maintenance immediately")

# =========================
# 5. SAVE RESULTS (if CSV input)
# =========================
if input_arg.endswith('.csv'):
    output_file = input_arg.replace('.csv', '_predictions.csv')
    
    # Combine input and predictions
    output_df = pd.concat([input_df, results_df], axis=1)
    output_df.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print("OUTPUT SAVED")
    print("="*80)
    print(f"✅ Predictions saved to: {output_file}")
    print(f"   Columns: {len(output_df.columns)} ({len(feature_names)} features + {len(target_names)} predictions)")

print("\n" + "="*80)
print("✅ PREDICTION COMPLETE")
print("="*80)