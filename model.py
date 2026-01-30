import pandas as pd
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, GroupKFold, cross_val_predict, GridSearchCV
from sklearn.metrics import classification_report, f1_score, make_scorer, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt

from feature_extraction import *

def model(features):

    df = pd.DataFrame(features)

    feature_names = [
        'Mean_PP', 'Std_PP', 'Mean_HR', 'Std_HR', 'SD2',
        'Mean_BVP', 'Median_BVP', 'Mode_BVP', 'Min_BVP', 'Max_BVP', 'Std_BVP',
        'M_d1', 'Std_d1', 'M_d2', 'Std_d2', 'HF',
        'Mean_EDA', 'Median_EDA', 'Mode_EDA', 'Max_EDA', 'Min_EDA', 'Std_EDA',
        'N_PEAKS', 'M_Amp', 'M_RT', 'M_D'
    ]
    

    X = df[feature_names].values
    y = df['label'].values
    groups = df['subject_id'].values

    pipeline = ImbPipeline([
       # ('smote', SMOTE(random_state=42)),
        ('classifier', BalancedRandomForestClassifier(

         #   class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    gkf = GroupKFold(n_splits=29)

    
    param_grid = {
        'classifier__n_estimators': [100],
        'classifier__max_depth': [ 8], 
        'classifier__min_samples_split': [10],
        'classifier__min_samples_leaf': [4]
    }
    
    gs = GridSearchCV(pipeline, param_grid, scoring="accuracy", cv=gkf)
    gs.fit(X, y, groups=groups)
    print(gs.best_params_, gs.best_score_)

    cv_results = cross_validate(
    gs.best_estimator_, 
    X, y, 
    groups=groups, 
    cv=gkf, 
    scoring={
        'accuracy': 'accuracy',
        'f1': 'f1',
        'precision': 'precision',
        'recall': 'recall'
    },
    return_train_score=True 
)   
    print(f"\nTraining Accuracy Mean: {np.mean(cv_results['train_accuracy']):.4f}")
    print(f"\nAccuracy Mean: {np.mean(cv_results['test_accuracy']):.4f}")
    print(f"F1-Score Mean: {np.mean(cv_results['test_f1']):.4f}")
    print(f"Precision Mean: {np.mean(cv_results['test_precision']):.4f}")
    print(f"Recall Mean: {np.mean(cv_results['test_recall']):.4f}")
    train_acc = np.mean(cv_results['train_accuracy'])
    test_acc = np.mean(cv_results['test_accuracy'])
    y_pred = cross_val_predict(gs.best_estimator_, X, y, groups=groups, cv=gkf)

    cm = confusion_matrix(y, y_pred)
    print(cm)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['REST', 'STRESS'])

    cm_display.plot()
    plt.show()
    return cv_results

if __name__ == "__main__":
    subjects_data = load_all_subjects()
    preprocessed_data = preprocess_subjects(subjects_data)
    all_segments, subject_durations = segmentation(preprocessed_data)
    all_labeled_segments = assign_labels(all_segments, subject_durations)
    valid_segments = extract_peak2peak(all_labeled_segments)
    features = extract_features(valid_segments)
    cv_scores = model(features)

