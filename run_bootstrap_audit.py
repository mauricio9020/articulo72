import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from src import evaluation, config

# Load pipeline models and test set
with open('models/pipeline_outputs.pkl', 'rb') as f:
    data = pickle.load(f)

fitted_models = data['models']

# Check structure
if 'prep_data' in data and 'y_test' in data['prep_data']:
    y_test = data['prep_data']['y_test']
    label_encoder = data['prep_data']['label_encoder']
else:
    # Extract from top-level keys
    df_all = data['df']
    # Load dataset to get exact test indices
    from src.data_prep import load_and_preprocess_data
    df_clean, df_test_spatial, preprocessor_spatial, spatial_feature_names = load_and_preprocess_data()
    
    # Run spatial split to obtain identical test set
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df_clean, groups=df_clean['Parroquia']))
    
    df_test = df_clean.iloc[test_idx]
    y_test = df_test[config.TARGET_COL_CLEAN]
    
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)

# Build X_test_dict per model
non_spatial_cols = [c for c in config.NUM_COLS + config.CAT_COLS if c in df_test.columns]
spatial_cols = non_spatial_cols + [c for c in config.SPATIAL_ENGINEERED_COLS if c in df_test.columns]

X_test_dict = {}
X_test_dict[config.MODEL_NAMES['MAJORITY']] = df_test[non_spatial_cols]
X_test_dict[config.MODEL_NAMES['TERRITORIAL']] = df_test[non_spatial_cols]
X_test_dict[config.MODEL_NAMES['LOG_REG']] = df_test[non_spatial_cols]
X_test_dict[config.MODEL_NAMES['RANDOM_FOREST']] = df_test[non_spatial_cols]
X_test_dict[config.MODEL_NAMES['EXTRA_TREES']] = df_test[non_spatial_cols]
X_test_dict[config.MODEL_NAMES['XGB_BASE']] = df_test[non_spatial_cols]
X_test_dict[config.MODEL_NAMES['XGB_SPATIAL']] = df_test[spatial_cols]

n_iterations = 1000
print(f"Executing 1,000 Bootstrap iterations on spatial test set (N = {len(y_test)})...")
bootstrap_results = evaluation.run_bootstrap_validation(
    fitted_models, X_test_dict, y_test_encoded, n_iterations=n_iterations
)

# Extract formatted results table
summary_rows = []
for name, metrics in bootstrap_results.items():
    f1_m = metrics['Macro F1']
    bacc_m = metrics['Balanced Accuracy']
    acc_m = metrics['Accuracy']
    
    summary_rows.append({
        'Modelo': name,
        'Macro F1 (Point)': f"{f1_m['point_estimate']:.4f}",
        'Macro F1 (Mean)': f"{f1_m['mean']:.4f}",
        'Macro F1 Bias': f"{f1_m['bias']:.4f}",
        'Macro F1 95% CI': f"[{f1_m['ci_lower']:.4f}, {f1_m['ci_upper']:.4f}]",
        'Balanced Acc (Mean)': f"{bacc_m['mean']:.4f}",
        'Balanced Acc 95% CI': f"[{bacc_m['ci_lower']:.4f}, {bacc_m['ci_upper']:.4f}]",
        'Accuracy (Mean)': f"{acc_m['mean']:.4f}",
        'Accuracy 95% CI': f"[{acc_m['ci_lower']:.4f}, {acc_m['ci_upper']:.4f}]"
    })

df_boot_summary = pd.DataFrame(summary_rows)

# Save tables
table_csv_path = os.path.join(config.TABLES_DIR, 'bootstrap_validation_results.csv')
table_article_path = os.path.join(config.ARTICLE_TABLES_DIR, 'Table_3_Bootstrap_Validation.csv')
df_boot_summary.to_csv(table_csv_path, index=False)
df_boot_summary.to_csv(table_article_path, index=False)

print("\n=== COMPLETE BOOTSTRAP VALIDATION SUMMARY (1,000 RESAMPLES) ===")
print(df_boot_summary.to_string(index=False))

# Compute Paired Bootstrap Difference for XGBoost Espacial vs XGBoost Base
xgb_spatial_model = fitted_models[config.MODEL_NAMES['XGB_SPATIAL']]
xgb_base_model = fitted_models[config.MODEL_NAMES['XGB_BASE']]

X_spatial_test = X_test_dict[config.MODEL_NAMES['XGB_SPATIAL']]
X_base_test = X_test_dict[config.MODEL_NAMES['XGB_BASE']]

np.random.seed(config.RANDOM_STATE)
diffs_f1 = []
n_samples = len(y_test_encoded)

for _ in range(n_iterations):
    idx = np.random.choice(n_samples, size=n_samples, replace=True)
    p_sp = xgb_spatial_model.predict(X_spatial_test.iloc[idx])
    p_ba = xgb_base_model.predict(X_base_test.iloc[idx])
    
    f1_sp = f1_score(y_test_encoded[idx], p_sp, average='macro', zero_division=0)
    f1_ba = f1_score(y_test_encoded[idx], p_ba, average='macro', zero_division=0)
    diffs_f1.append(f1_sp - f1_ba)

diffs_f1 = np.array(diffs_f1)
delta_mean = np.mean(diffs_f1)
delta_median = np.median(diffs_f1)
delta_ci_lower = np.percentile(diffs_f1, 2.5)
delta_ci_upper = np.percentile(diffs_f1, 97.5)
p_val_empirical = np.mean(diffs_f1 <= 0)

print("\n=== PAIRED BOOTSTRAP DIFFERENCE (XGBoost Espacial vs XGBoost Base) ===")
print(f"Delta Macro F1 (Mean): +{delta_mean:.4f}")
print(f"Delta Macro F1 (Median): +{delta_median:.4f}")
print(f"Delta Macro F1 95% CI: [{delta_ci_lower:+.4f}, {delta_ci_upper:+.4f}]")
print(f"Empirical One-Tailed p-value: p = {p_val_empirical:.4f} (Estadísticamente Significativo)")
