import pandas as pd
import numpy as np
import os
import math 
from scipy.signal import resample_poly, filtfilt
from scipy.signal.windows import gaussian
 
def eda_conversion(signal):
    target_fs = 64
    eda_fs = 4.0
    eda_upsampled = resample_poly(signal, target_fs, int(eda_fs))
    signal = eda_upsampled
    fs = 64
    total_duration_minutes = len(signal) / fs / 60

    first_segment_end_index = round(27 * 60 * fs)
    last_segment_start_index = round((total_duration_minutes - 5) * 60 * fs)

    first_segment = signal[:first_segment_end_index]
    last_segment = signal[last_segment_start_index:]
    middle_segment = signal[first_segment_end_index:last_segment_start_index] 
    #because the math task is limitless

    num_whole_minutes = math.floor(len(middle_segment) / fs / 60)

    first_matrix = []
    middle_matrix = []
    last_matrix = []

    for i in range(27): 
        start_index = i * fs * 60
        end_index = (i + 1) * fs * 60
        if end_index <= len(first_segment):
            first_matrix.append(first_segment[start_index:end_index])

    for i in range(num_whole_minutes): 
        segment_duration = fs * 60  
        start_index = i * segment_duration
        end_index = (i + 1) * segment_duration
        if end_index <= len(middle_segment):
            middle_matrix.append(middle_segment[start_index:end_index])

    for i in range(5):
        start_index = i * fs * 60
        end_index = (i + 1) * fs * 60
        if end_index <= len(last_segment):
            last_matrix.append(last_segment[start_index : end_index ])

    all_segments = np.vstack([first_matrix, middle_matrix, last_matrix])
    matrix_length = len(all_segments) 

    pattern = np.ones(matrix_length, dtype=int)
    pattern[[0, 1, 2, 13, 14, 20, 21, 25, 26]] = 0 #Rest 

    pattern[-5:] = [0, 0, 1, 0, 0]

    tag_matrix = pattern
    numSegments = len(all_segments)
    percent_removed_all = np.zeros(numSegments)
    N = 40
    alpha=0.4


    filtered_segments = []
    for j in range(numSegments):
        current_segment = all_segments[j, :]

        std = (N - 1) / (2 * alpha * np.sqrt(2 * np.log(2)))
        gauss_filter = gaussian(N, std)
        gauss_filter = gauss_filter / np.sum(gauss_filter)  

        # Apply filter
        eda_smoothed = filtfilt(gauss_filter, 1, current_segment)
        filtered_segments.append(eda_smoothed)
   # print(" EDA conversion results")

   # print(f"Total duration: {total_duration_minutes:.2f} minutes")
    #print(f"Total segments: {numSegments}")

    #print(f"  REST (0): {np.sum(tag_matrix == 0)}")
    #print(f"  STRESS (1): {np.sum(tag_matrix == 1)}")

    return np.array(filtered_segments), tag_matrix