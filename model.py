import pandas as pd
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
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
   
        
        # Mean and std of each feature during REST only
        rest_means = rest_segments[feature_names].mean()
        rest_stds = rest_segments[feature_names].std(ddof=1)
        
        # If a feature has zero variance at rest, don't divide by zero
        rest_stds = rest_stds.replace(0, 1)
        
        # Normalize all segments relative to rest baseline
        subject_normalized = subject_df.copy()
        subject_normalized[feature_names] = (subject_df[feature_names] - rest_means) / rest_stds
        
        normalized_dfs.append(subject_normalized)
    #
    return pd.concat(normalized_dfs, ignore_index=True)

def model(features):

    df = pd.DataFrame(features)

    feature_names  = [
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

       # ('smote', SMOTE(random_state=42)),
        ('classifier', BalancedRandomForestClassifier(

         #   class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    gkf = GroupKFold(n_splits=29)
 
    
    '''param_grid  = {
    'pca__n_components':            [10, 15, 20, 25],
    'classifier__n_estimators':     [100, 200, 300],
    'classifier__max_depth':        [5, 6, 8, 10, 15, 20],
    'classifier__min_samples_split':[5, 8, 10, 15, 20],
    'classifier__min_samples_leaf': [2, 4, 6, 8],
}'''
    param_grid = {
        'classifier__n_estimators':      [100, 200, 300],
        'classifier__max_depth':         [10, 15, 20],
        'classifier__min_samples_split': [10, 15, 20],
        'classifier__min_samples_leaf':  [2, 4],
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

    return_train_score=True )
    print(f"\nTraining Accuracy Mean: {np.mean(cv_results['train_accuracy']):.4f}")
    print(f"Accuracy Mean: {np.mean(cv_results['test_accuracy']):.4f}")
    print(f"Balanced_Accuracy Mean: {np.mean(cv_results['test_balanced_accuracy']):.4f}")
    print(f"F1-Score Mean: {np.mean(cv_results['test_f1']):.4f}")
    print(f"Precision Mean: {np.mean(cv_results['test_precision']):.4f}")
    print(f"Recall Mean: {np.mean(cv_results['test_recall']):.4f}")
    train_acc = np.mean(cv_results['train_accuracy'])
    test_acc = np.mean(cv_results['test_accuracy'])
    y_pred = cross_val_predict(gs.best_estimator_, X, y, groups=groups, cv=gkf)

    cm = confusion_matrix(y, y_pred)
    print(cm)
    #cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['REST', 'STRESS'])

    #cm_display.plot()
    #plt.show()
    false_negatives = np.where((y == 1) & (y_pred == 0))[0]  # missed stress
    false_positives = np.where((y == 0) & (y_pred == 1))[0]  # wrong stress prediction

   # print("False negative indices:", false_negatives)
   # print("False positive indices:", false_positives)
   # print(df.iloc[false_negatives][['subject_id', 'seg_start', 'label']])
    fn_df = df.iloc[false_negatives][['subject_id', 'seg_start', 'label']].copy()
    fp_df = df.iloc[false_positives][['subject_id', 'seg_start', 'label']].copy()

    fn_df['seg_start_min'] = fn_df['seg_start'] / 60
    fp_df['seg_start_min'] = fp_df['seg_start'] / 60

  #  print("False negatives by minute:")
  #  print(fn_df['seg_start_min'].value_counts().sort_index())

   # print("\nFalse positives by minute:")
   # print(fp_df['seg_start_min'].value_counts().sort_index())
    return cv_results, gs.best_estimator_, X, y, groups, gkf

if __name__ == "__main__":
    subjects_data = load_all_subjects()
    preprocessed_data = preprocess_subjects(subjects_data)
    all_segments, subject_durations = segmentation(preprocessed_data)
    all_labeled_segments = assign_labels(all_segments, subject_durations)
    valid_segments = extract_peak2peak(all_labeled_segments)
    features = extract_features(valid_segments)
    cv_scores = model(features)

