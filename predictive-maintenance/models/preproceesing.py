import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ROBUST PREPROCESSING FOR PREDICTIVE MAINTENANCE")
print("="*80)

# =========================
# 1. LOAD DATA
# =========================
print("\n[1/8] Loading data...")
df = pd.read_csv('../data/merged_dataset.csv')
print(f"✅ Original shape: {df.shape}")
print(f"   Columns: {df.shape[1]}")
print(f"   Rows: {df.shape[0]}")

# =========================
# 2. CLEAN COLUMN NAMES
# =========================
print("\n[2/8] Cleaning column names...")
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)
print(f"✅ Column names cleaned")

# Remove exact duplicates
initial_rows = len(df)
df = df.drop_duplicates()
print(f"   Removed {initial_rows - len(df)} duplicate rows")
print(f"   Current shape: {df.shape}")

# =========================
# 3. IDENTIFY COLUMNS
# =========================
print("\n[3/8] Identifying columns...")

# Target columns (what we want to predict)
target_cols = ['vibration_index', 'thermal_index', 'efficiency_index']
target_cols = [col for col in target_cols if col in df.columns]
print(f"✅ Target columns: {target_cols}")

# Columns to drop (non-predictive or redundant)
drop_cols = [
    'engine_id', 'udi', 'product_id', 'machine_failure', 
    'twf', 'hdf', 'pwf', 'osf', 'rnf',
    'overall_hi', 'hi_smooth', 'hi_trend', 'risk', 'fuel_index',
    'datetime'  # timestamps
]
drop_cols = [col for col in drop_cols if col in df.columns]
print(f"   Dropping: {drop_cols}")

# Feature columns
feature_cols = [col for col in df.columns if col not in target_cols + drop_cols]
print(f"✅ Feature columns: {len(feature_cols)}")

# =========================
# 4. PREPARE FEATURES
# =========================
print("\n[4/8] Preparing features...")

X = df[feature_cols].copy()
print(f"   Initial features: {X.shape}")

# Handle categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"   Categorical columns: {categorical_cols}")

for col in categorical_cols:
    print(f"      Encoding {col}...")
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Convert all to numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

print(f"✅ All features converted to numeric")

# =========================
# 5. FEATURE ENGINEERING
# =========================
print("\n[5/8] Engineering features...")

original_feature_count = X.shape[1]

# Temperature features
if 'process_temperature_k' in X.columns and 'air_temperature_k' in X.columns:
    X['temp_difference'] = X['process_temperature_k'] - X['air_temperature_k']
    X['temp_ratio'] = X['process_temperature_k'] / (X['air_temperature_k'] + 1e-6)

# Mechanical features
if 'torque_nm' in X.columns and 'rotational_speed_rpm' in X.columns:
    X['mechanical_power'] = (X['torque_nm'] * X['rotational_speed_rpm'] * 2 * np.pi) / 60000
    X['mechanical_stress'] = X['torque_nm'] * X['rotational_speed_rpm']
    X['torque_per_speed'] = X['torque_nm'] / (X['rotational_speed_rpm'] + 1e-6)

# Tool wear features
if 'tool_wear_min' in X.columns:
    X['tool_wear_squared'] = X['tool_wear_min'] ** 2
    X['tool_wear_log'] = np.log1p(X['tool_wear_min'])

# Environmental features
if 'humidity' in X.columns and 'temperature' in X.columns:
    X['humidity_temp_interaction'] = X['humidity'] * X['temperature']

if 'rainfall' in X.columns:
    X['rainfall_binary'] = (X['rainfall'] > 0).astype(int)

# Cycle-based features
if 'cycle' in X.columns:
    X['cycle_squared'] = X['cycle'] ** 2

print(f"✅ Added {X.shape[1] - original_feature_count} engineered features")
print(f"   Total features now: {X.shape[1]}")

# =========================
# 6. HANDLE MISSING AND INVALID VALUES
# =========================
print("\n[6/8] Handling missing and invalid values...")

# Check for NaN
nan_count_before = X.isna().sum().sum()
print(f"   NaN values before: {nan_count_before}")

# Replace infinite values with NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Fill NaN with median
X = X.fillna(X.median())

# If still NaN (column was all NaN), fill with 0
X = X.fillna(0)

nan_count_after = X.isna().sum().sum()
print(f"✅ NaN values after: {nan_count_after}")

# =========================
# 7. PREPARE TARGETS
# =========================
print("\n[7/8] Preparing targets...")

y = df[target_cols].copy()
print(f"   Target shape: {y.shape}")
print(f"\n   Target statistics:")
print(y.describe())

# Handle missing targets
y = y.fillna(y.median())
print(f"✅ Targets cleaned")

# =========================
# 8. TRAIN-TEST SPLIT (80-20)
# =========================
print("\n[8/8] Splitting data (80-20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"✅ Split complete:")
print(f"   Train set: X={X_train.shape}, y={y_train.shape}")
print(f"   Test set:  X={X_test.shape}, y={y_test.shape}")

# Verify we have data
if len(X_train) == 0:
    print("\n❌ ERROR: Training set is empty!")
    print("   This should not happen. Check your data.")
    exit(1)

# =========================
# 9. SCALING
# =========================
print("\n[9/9] Scaling features...")

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print(f"✅ Scaling complete")

# =========================
# 10. OPTIONAL: REMOVE EXTREME OUTLIERS (CAREFULLY!)
# =========================
print("\n[10/10] Checking for extreme outliers...")

# Calculate z-scores
z_scores = np.abs(stats.zscore(X_train_scaled, nan_policy='omit'))

# Count outliers per sample
outlier_counts = (z_scores > 5).sum(axis=1)  # Using 5 std dev (very conservative)

# Only remove samples with many outliers (>50% of features are outliers)
outlier_threshold = X_train_scaled.shape[1] * 0.5
extreme_outliers = outlier_counts > outlier_threshold

num_outliers = extreme_outliers.sum()
print(f"   Found {num_outliers} extreme outliers (>{outlier_threshold:.0f} features > 5σ)")

if num_outliers > 0 and num_outliers < len(X_train_scaled) * 0.2:  # Don't remove more than 20%
    X_train_scaled = X_train_scaled[~extreme_outliers]
    y_train = y_train[~extreme_outliers]
    print(f"   Removed {num_outliers} extreme outliers")
    print(f"✅ Final training shape: X={X_train_scaled.shape}, y={y_train.shape}")
else:
    print(f"   No extreme outliers removed (would remove too many samples)")

# =========================
# 11. SAVE FILES
# =========================
print("\n" + "="*80)
print("SAVING PREPROCESSED FILES")
print("="*80)

X_train_scaled.to_csv('X_train.csv', index=False)
X_test_scaled.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print(f"\n✅ Files saved successfully:")
print(f"   - X_train.csv: {X_train_scaled.shape}")
print(f"   - X_test.csv:  {X_test_scaled.shape}")
print(f"   - y_train.csv: {y_train.shape}")
print(f"   - y_test.csv:  {y_test.shape}")

print(f"\n📊 Summary:")
print(f"   Total features: {X_train_scaled.shape[1]}")
print(f"   Target columns: {target_cols}")
print(f"   Train samples: {len(X_train_scaled)}")
print(f"   Test samples: {len(X_test_scaled)}")

# Sanity checks
print("\n🔍 Data Quality Checks:")
print(f"   ✅ No NaN in X_train: {not X_train_scaled.isna().any().any()}")
print(f"   ✅ No NaN in X_test: {not X_test_scaled.isna().any().any()}")
print(f"   ✅ No NaN in y_train: {not y_train.isna().any().any()}")
print(f"   ✅ No NaN in y_test: {not y_test.isna().any().any()}")
print(f"   ✅ Train set not empty: {len(X_train_scaled) > 0}")
print(f"   ✅ Test set not empty: {len(X_test_scaled) > 0}")

print("\n" + "="*80)
print("✅ PREPROCESSING COMPLETE - READY FOR TRAINING!")
print("="*80)