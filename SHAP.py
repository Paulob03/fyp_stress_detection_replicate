import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

from load_data import *
from preprocess import *
from segmentation import *
from feature_extraction import *
from model import subject_normalize

WINDOW   = 60
STEP     = 20
OUT_DIR  = "shap_output"

import os; os.makedirs(OUT_DIR, exist_ok=True)

feature_names = [
    'Mean_PP', 'Std_PP', 'Mean_HR', 'Std_HR', 'SD2', 'RMSSD', 'SD1', 'pNN50',
    'Mean_BVP', 'Median_BVP', 'Mode_BVP', 'Min_BVP', 'Max_BVP', 'Std_BVP',
    'M_d1', 'Std_d1', 'M_d2', 'Std_d2', 'HF',
    'Mean_EDA', 'Median_EDA', 'Mode_EDA', 'Max_EDA', 'Min_EDA', 'Std_EDA',
    'N_PEAKS', 'M_Amp', 'M_RT', 'M_D', 'LF', 'LF_HF_Ratio', 'HFn'
]



subjects_data     = load_all_subjects()
preprocessed_data = preprocess_subjects(subjects_data)
all_segments      = sliding_windows(preprocessed_data, segment_duration=WINDOW, step_size=STEP)
valid_segments    = extract_peak2peak(all_segments)
features          = extract_features(valid_segments)

df     = subject_normalize(features, feature_names)
X      = df[feature_names].values
y      = df['label'].values
groups = df['subject_id'].values


pipeline = ImbPipeline([
    ('scaler',     StandardScaler()),
    ('classifier', BalancedRandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    ))
])


pipeline.fit(X, y)
rf_model = pipeline.named_steps['classifier']

# Transform X through scaler only (so SHAP sees scaled but not PCA'd features)
X_scaled = pipeline.named_steps['scaler'].transform(X)

# SHAP

explainer   = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_scaled)

# Handle both old (list) and new (3D array) SHAP output formats
if isinstance(shap_values, list):
    sv_stress = shap_values[1]       # old format: list[class0, class1]
else:
    sv_stress = shap_values[:, :, 1] # new format: (samples, features, classes)

# ── Plot 1: Beeswarm - feature importance + direction ─────────────────────────

shap.summary_plot(
    sv_stress, X_scaled,
    feature_names=feature_names,
    show=False
)
plt.title(f"SHAP Beeswarm - Stress class ({WINDOW}s window)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shap_beeswarm_win{WINDOW}.png", dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 2: Bar chart - mean absolute SHAP (overall importance) ───────────────
print("Saving bar plot...")
shap.summary_plot(
    sv_stress, X_scaled,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
plt.title(f"SHAP Feature Importance ({WINDOW}s window)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shap_bar_win{WINDOW}.png", dpi=150, bbox_inches='tight')
plt.close()

# ── Print top 10 features by mean absolute SHAP ───────────────────────────────
mean_abs_shap = np.abs(sv_stress).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)

print("\nTop 10 features by SHAP importance:")
print(importance_df.head(10).to_string(index=False))
importance_df.to_csv(f"{OUT_DIR}/shap_importance_win{WINDOW}.csv", index=False)
print(f"\nAll outputs saved to {OUT_DIR}/")