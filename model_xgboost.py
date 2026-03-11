import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, GroupKFold, cross_val_predict, GridSearchCV
from sklearn.metrics import classification_report, f1_score, make_scorer, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from feature_extraction import *

def subject_normalize(features, feature_names):

    df = pd.DataFrame(features)
    
    normalized_dfs = []
    
    for subject_id, subject_df in df.groupby('subject_id'):
        rest_segments = subject_df[subject_df['label'] == 0]
        
        rest_means = rest_segments[feature_names].mean()
        rest_stds = rest_segments[feature_names].std(ddof=1)
        
        rest_stds = rest_stds.replace(0, 1)
        
        subject_normalized = subject_df.copy()
        subject_normalized[feature_names] = (subject_df[feature_names] - rest_means) / rest_stds
        
        normalized_dfs.append(subject_normalized)

    return pd.concat(normalized_dfs, ignore_index=True)

def model(features):

    df = pd.DataFrame(features)

    feature_names = [
        'Mean_PP', 'Std_PP', 'Mean_HR', 'Std_HR', 'SD2', 'RMSSD', 'SD1', 'pNN50',
        'Mean_BVP', 'Median_BVP', 'Mode_BVP', 'Min_BVP', 'Max_BVP', 'Std_BVP',
        'M_d1', 'Std_d1', 'M_d2', 'Std_d2', 'HF',
        'Mean_EDA', 'Median_EDA', 'Mode_EDA', 'Max_EDA', 'Min_EDA', 'Std_EDA',
        'N_PEAKS', 'M_Amp', 'M_RT', 'M_D', 'LF', 'LF_HF_Ratio', 'HFn',
        'SCL_Mean', 'SCL_Slope', 'ISCR', 'Phasic_Std',
        'M_SCR', 'HR_EDA_Corr', 'BVP_EDA_Var_Ratio'
    ]

    df = subject_normalize(features, feature_names)
    X = df[feature_names].values
    y = df['label'].values
    groups = df['subject_id'].values

    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),

        ('classifier', XGBClassifier(
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            verbosity=0
        ))
    ])

    gkf = GroupKFold(n_splits=29)
    
    param_grid = {
    'classifier__n_estimators':  [100, 200, 300],
    'classifier__max_depth':     [5, 6, 8, 10, 15, 20],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__subsample':     [0.8, 1.0],
}

    gs = GridSearchCV(pipeline, param_grid, scoring="balanced_accuracy", cv=gkf, n_jobs=-1)
    gs.fit(X, y, groups=groups)
    print(gs.best_params_, gs.best_score_)

    cv_results = cross_validate(
        gs.best_estimator_,
        X, y,
        groups=groups,
        cv=gkf,
        scoring={
            'accuracy': 'accuracy',
            'balanced_accuracy': 'balanced_accuracy',
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall'
        },
        return_train_score=True
    )

    print(f"\nTraining Accuracy Mean: {np.mean(cv_results['train_accuracy']):.4f}")
    print(f"Accuracy Mean: {np.mean(cv_results['test_accuracy']):.4f}")
    print(f"Balanced_Accuracy Mean: {np.mean(cv_results['test_balanced_accuracy']):.4f}")
    print(f"F1-Score Mean: {np.mean(cv_results['test_f1']):.4f}")
    print(f"Precision Mean: {np.mean(cv_results['test_precision']):.4f}")
    print(f"Recall Mean: {np.mean(cv_results['test_recall']):.4f}")

    y_pred = cross_val_predict(gs.best_estimator_, X, y, groups=groups, cv=gkf)
    cm = confusion_matrix(y, y_pred)
    print(cm)

    fn_df = df.iloc[np.where((y == 1) & (y_pred == 0))[0]][['subject_id', 'seg_start', 'label']].copy()
    fp_df = df.iloc[np.where((y == 0) & (y_pred == 1))[0]][['subject_id', 'seg_start', 'label']].copy()

    fn_df['seg_start_min'] = fn_df['seg_start'] / 60
    fp_df['seg_start_min'] = fp_df['seg_start'] / 60

    return cv_results, gs.best_estimator_, X, y, groups, gkf

if __name__ == "__main__":
    print("Pauls pipeline and model\n")
    subjects_data = load_all_subjects()
    preprocessed_data = preprocess_subjects(subjects_data)
    #all_segments, subject_durations = segmentation(preprocessed_data)
    all_labeled_segments = sliding_windows(preprocessed_data)
  #  all_labeled_segments = assign_labels(all_segments, subject_durations)
    #all_labeled_segments = sliding_window_assign_labels(all_segments, subject_durations, segment_duration, step_size)
    valid_segments = extract_peak2peak(all_labeled_segments)
    features = extract_features(valid_segments)
    cv_scores = model(features)