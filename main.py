from load_data import *
from preprocess import *
from segmentation import *
from feature_extraction import *
from model import *
from RFwith10CV import *
from matlab_conversion.PPG_conversion import run_all_subjects
from matlab_conversion.EDA_conversion import *

def main():
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

    

    #print("Matlab pipeline and paper model\n")
    #all_segments = run_all_subjects()
    #features1 = extract_features(all_segments)
    #paper_model(features1)

main()