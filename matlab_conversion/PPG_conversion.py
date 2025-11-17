import pandas as pd
import numpy as np
import os
import math 
from scipy.signal import cheby2, sosfiltfilt, find_peaks, welch
import matplotlib.pyplot as plt
from matlab_conversion.EDA_conversion import *
from biosppy.signals import eda as eda_module
from scipy import stats
from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.metrics import classification_report, f1_score, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier


DATA_DIR = 'Subjects'
subject_id = 'subject_01'
signal = pd.read_csv(os.path.join(DATA_DIR, subject_id, 'BVP.csv'), header=None).values.flatten()



def ppg_conversion(signal):
    #DATA_DIR = 'Subjects'
    #subject_id = 'subject_01'
    #signal = pd.read_csv(os.path.join(DATA_DIR, subject_id, 'BVP.csv'), header=None).values.flatten()

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
    order = 2
    rs = 20
    low_freq = 0.5
    high_freq = 5.0
    bvp_fs = 64.0
    nyquist = bvp_fs / 2
    low_normalized = low_freq / nyquist
    high_normalized = high_freq / nyquist

    min_peak_distance_sec = 0.4
    min_peak_distance_samples = int(min_peak_distance_sec * bvp_fs)
    min_peak_height = 0
    percent_removed_all = np.zeros(numSegments)

    all_peaks = []
    all_pp_intervals = []

    for j in range(numSegments):
        current_segment = all_segments[j, :]

        fcutlow =0.5
        fcuthigh = 5
        sos = cheby2(order, rs, [low_normalized, high_normalized], btype='bandpass', output='sos')
        filtered_PPG = sosfiltfilt(sos, current_segment)
        peaks, _ = find_peaks(filtered_PPG, distance=min_peak_distance_samples, height=min_peak_height)
        rpi = np.diff(peaks) / fs 
        original = len(rpi)


        valid_mask = (rpi >= 0.5) & (rpi <= 1.2)
        valid_intervals = rpi[valid_mask]
        all_peaks.append(peaks)  
        all_pp_intervals.append(valid_intervals) 

        modified = len(valid_intervals)

        if original > 0:
            percent_removed = (original - modified) / original * 100
        else:
            percent_removed = 100 
        
        percent_removed_all[j] = percent_removed

    segments_percentage_tag = np.column_stack([all_segments, tag_matrix, percent_removed_all])
    rows_toKeep = segments_percentage_tag[:, -1] < 15
    segments_percentage_tag = segments_percentage_tag[rows_toKeep, :]


    segments_after_rejecting = segments_percentage_tag[:, :3840]  # 60 sec * 64 Hz = 3840 samples
    tags_after_rejecting = segments_percentage_tag[:, 3840]  # Labels: 0=REST, 1=STRESS

    num_segments_after = len(segments_after_rejecting)
    for j in range(num_segments_after):
        current_segment = segments_after_rejecting[j, :]
        
        fcutlow = 0.5
        fcuthigh = 5
        sos = cheby2(2, 20, [fcutlow / nyquist, fcuthigh / nyquist], btype='bandpass', output='sos')
        filtered_PPG = sosfiltfilt(sos, current_segment)



   # print(" PPG conversion Results")
   # print(f"Total duration: {total_duration_minutes:.2f} minutes")
   # print(f"Total segments before QC: {numSegments}")
   # print(f"Segments after QC (< 15% removed): {num_segments_after}")
   # print(f"Rejection rate: {(numSegments - num_segments_after)/numSegments*100:.1f}%")
   # print(f"  REST (0): {np.sum(tags_after_rejecting == 0)}")
   # print(f"  STRESS (1): {np.sum(tags_after_rejecting == 1)}")



    return segments_after_rejecting, tags_after_rejecting, rows_toKeep, all_peaks, all_pp_intervals


def load_subject_signals(subject_folder):
    bvp_df = pd.read_csv(os.path.join(subject_folder, 'BVP.csv'), header=None)
    bvp_signal = bvp_df.iloc[:, 0].astype(float).values
    eda_df = pd.read_csv(os.path.join(subject_folder, 'EDA.csv'), header=None)
    eda_df[0] = eda_df[0].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    eda_signal = eda_df.iloc[:, 0].astype(float).values

    return bvp_signal, eda_signal


def run_all_subjects():
    DATA_DIR = "Subjects"
    all_segments = []
    count = 0
    for subject_num in range(1, 30):
        subject_id = f"subject_{subject_num:02d}"
       # print(f"subject_{subject_num:02d}")
        folder = os.path.join(DATA_DIR, subject_id)
        
        bvp, eda = load_subject_signals(folder)
        ppg_segments, ppg_labels, rows_to_keep, peaks_list, pp_list = ppg_conversion(bvp)
        eda_segments, eda_labels = eda_conversion(eda)
        if len(rows_to_keep) == len(eda_segments):
            eda_segments_matched = eda_segments[rows_to_keep]
            count += 1
            #print(f"subjects: {count}")
            for i in range(len(ppg_segments)):
                all_segments.append({
                    'bvp_segment': ppg_segments[i],
                    'eda_segment': eda_segments_matched[i],
                    'label': int(ppg_labels[i]),
                    'subject_id': subject_id,
                    'segment_idx': i,
                    'peaks': peaks_list[i],  
                    'pp_intervals': pp_list[i] 
                })

        else:
            pass
        
    return all_segments 


def extract_features(valid_segments):


    for segment in valid_segments:
        bvp_signal = segment['bvp_segment']
        eda_signal = segment['eda_segment']
        pp_intervals = segment['pp_intervals']

        #Mean peak-to-peak interval (seconds)
        segment['Mean_PP'] = np.mean(pp_intervals)

        #Standard deviation of PP intervals
        segment['Std_PP'] = np.std(pp_intervals)

        #Mean heart rate (BPM): 60 / mean(PP)
        segment['Mean_HR'] = 60 / np.mean(pp_intervals)

        #Standard deviation of heart rate
        segment['Std_HR'] = np.std(60 / pp_intervals)
        
        #Poincaré plot standard deviation: √0.5 × std(PP[i] + PP[i+1])
        pp_sum = pp_intervals[1:] + pp_intervals[:-1]
        segment['SD2'] = np.sqrt(0.5) * np.std(pp_sum)

        #Mean BVP
        segment['Mean_BVP'] = np.mean(bvp_signal)
        #Meadian BVP
        segment['Median_BVP'] = np.median(bvp_signal)


        mode_result = stats.mode(np.round(bvp_signal, 1), keepdims=True)
        segment['Mode_BVP'] = float(mode_result.mode[0])

        segment['Min_BVP'] = np.min(bvp_signal)
        segment['Max_BVP'] = np.max(bvp_signal)
        segment['Std_BVP'] = np.std(bvp_signal)

        bvp_d1 = np.diff(bvp_signal)

        #rate of change
        segment['M_d1'] = np.mean(bvp_d1)
        segment['Std_d1'] = np.std(bvp_d1)

        #rate of acceleration
        bvp_d2 = np.diff(bvp_d1)
        segment['M_d2'] = np.mean(bvp_d2)
        segment['Std_d2'] = np.std(bvp_d2)

        #Mean skin conductance level
        segment['Mean_EDA'] = np.mean(eda_signal)
        #Median skin conductance level
        segment['Median_EDA'] = np.median(eda_signal)

        #Most frequent value of skin conductance level
        mode_result = stats.mode(np.round(eda_signal, 2), keepdims=True)
        segment['Mode_EDA'] = float(mode_result.mode[0])

        segment['Max_EDA'] = np.max(eda_signal)
        segment['Min_EDA'] = np.min(eda_signal)
        segment['Std_EDA'] = np.std(eda_signal)

        if len(pp_intervals) > 10:  

            time_pp = np.cumsum(pp_intervals)
            time_pp = np.insert(time_pp, 0, 0)  
            
        
            fs_resample = 4.0
            time_interp = np.arange(0, time_pp[-1], 1/fs_resample)
            

            pp_interp = np.interp(time_interp, time_pp[:-1], pp_intervals)
            

            freqs, psd = welch(pp_interp, fs=fs_resample, nperseg=min(256, len(pp_interp)))
            
            # HF band: 0.15 - 0.4 Hz
            hf_band = (freqs >= 0.15) & (freqs <= 0.4)
            #High-frequency power (0.15-0.4 Hz) from Welch PSD analysis*
            segment['HF'] = np.sum(psd[hf_band])
        else:
            segment['HF'] = 0

        try:
            eda_results = eda_module.eda(signal=eda_signal, sampling_rate=64.0, show=False)
            #Number of SCR peaks detected
            segment['N_PEAKS'] = len(eda_results['peaks'])
            #NMean SCR amplitude 
            segment['M_Amp'] = np.mean(eda_results['amplitudes'])
            #Mean rise time (seconds)
            segment['M_RT'] = np.mean(eda_results['rise_times'])

            rise_times = eda_results['rise_times']
            half_rec = eda_results['half_rec']
            valid_durations = []
            for i in range(len(rise_times)):
                if half_rec[i] is not None:
                    duration = rise_times[i] + half_rec[i]
                    valid_durations.append(duration)

            #Mean SCR duration (rise time + half-recovery time)
            if len(valid_durations) > 0:
                segment['M_D'] = np.mean(valid_durations)
            else:
                
                segment['M_D'] = segment['M_RT'] * 2.5 


            #segment['M_SCR'] = np.mean(eda_results['amplitudes'])
        except ValueError as e:
            #No SCR pulses detected in this segment
            segment['N_PEAKS'] = 0
            segment['M_Amp'] = 0
            segment['M_RT'] = 0
            segment['M_D'] = 0
    
    return valid_segments

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
    all_segments = run_all_subjects()
    features = extract_features(all_segments)
    cv_scores = model(features)