import pandas as pd
import numpy as np
import os
from scipy.signal import cheby2, filtfilt, find_peaks, resample_poly
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from load_data import *

#Loops through each subject to preprocess
def preprocess_subjects(subjects_data):
    preprocessed_subjects = []
    for subject in subjects_data:
        bvp_filtered = preprocess_bvp(subject['bvp_signal'])
        eda_processed = preprocess_eda(subject['eda_signal'])
        
        preprocessed_subjects.append({
            'subject_id': subject['subject_id'],
            'bvp_filtered': bvp_filtered,
            'eda_processed': eda_processed
        })
    #List of dictionaries with preprocessed signals
    return preprocessed_subjects


#bandpass filtering to BVP signal to isolate heart rate frequencies.
def preprocess_bvp(bvp_signal):
    order = 4
    rs = 20
    low_freq = 0.5
    high_freq = 5.0
    bvp_fs = 64.0
    nyquist = bvp_fs / 2
    low_normalized = low_freq / nyquist
    high_normalized = high_freq / nyquist

    b, a = cheby2(order, rs, [low_normalized, high_normalized], btype='bandpass')

    bvp_signal_filtered = filtfilt(b, a, bvp_signal)

    return bvp_signal_filtered

#Upsamples and smooths EDA signal to match BVP sampling rate
def preprocess_eda(eda_signal):
    target_fs = 64
    eda_fs = 4.0
    sigma_seconds = 0.4
    sigma_samples = sigma_seconds * target_fs
    eda_upsampled = resample_poly(eda_signal, target_fs, int(eda_fs))
    eda_smoothed = gaussian_filter1d(eda_upsampled, sigma=sigma_samples)

    return eda_smoothed






if __name__ == "__main__":
    subjects_data = load_all_subjects()
    preprocessed_data = preprocess_subjects(subjects_data)

