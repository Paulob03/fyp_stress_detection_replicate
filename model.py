import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.metrics import classification_report, f1_score, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

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
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ])
    #Before smote
    '''
    rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
    )'''

    gkf = GroupKFold(n_splits=10)



    cv_results = cross_validate(
        pipeline, X, y, 
        groups=groups, 
        cv=gkf, 
        scoring={
        'accuracy': 'accuracy',
        'f1': 'f1',
        'precision': 'precision',
        'recall': 'recall'},
        return_train_score=False
    )

    print("Accuracy Mean: " + str(np.mean(cv_results['test_accuracy'])))
    print("Accuracy Std: " + str(np.std(cv_results['test_accuracy'])))
    print("F1-Score Mean: " + str(np.mean(cv_results['test_f1'])))
    print("F1-Score Std: " + str(np.std(cv_results['test_f1'])))
    print("Precision Mean: " + str(np.mean(cv_results['test_precision'])))
    print("Precision Std: " + str(np.std(cv_results['test_precision'])))
    print("Recall Mean: " + str(np.mean(cv_results['test_recall'])))
    print("Recall Std: " + str(np.std(cv_results['test_recall'])))

    return cv_results

if __name__ == "__main__":
    subjects_data = load_all_subjects()
    preprocessed_data = preprocess_subjects(subjects_data)
    all_segments, subject_durations = segmentation(preprocessed_data)
    all_labeled_segments = assign_labels(all_segments, subject_durations)
    valid_segments = extract_peak2peak(all_labeled_segments)
    features = extract_features(valid_segments)
    cv_scores = model(features)

