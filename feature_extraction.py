import pandas as pd
import numpy as np
from preprocess import *
from scipy.signal import find_peaks
from scipy import stats
from biosppy.signals import eda as eda_module
from segmentation import *
from scipy.signal import welch


#Detects peaks in BVP signals and computes peak to peak intervals and filters out segments with too high abnormal percentage
def extract_peak2peak(all_labeled_segments):
    bvp_fs = 64.0
    min_peak_distance_sec = 0.4
    min_peak_distance_samples = int(min_peak_distance_sec * bvp_fs)
    min_peak_height = 0

    min_pp_interval = 0.5
    max_pp_interval = 1.2
    max_abnormal_percentage = 30

    
    for segment in all_labeled_segments:
        bvp_signal = segment['bvp_segment']

        peaks, _ = find_peaks(bvp_signal, distance=min_peak_distance_samples, height=min_peak_height)

        pp_intervals = np.diff(peaks) / bvp_fs

        segment['peaks'] = peaks

        segment['pp_intervals'] = pp_intervals

    
    valid_segments = []
    for segment in all_labeled_segments:
        pp_intervals = segment['pp_intervals']
        valid_mask = (pp_intervals >= min_pp_interval) & (pp_intervals <= max_pp_interval)
        clean_pp = pp_intervals[valid_mask]
        abnormal_pct = 100 * (1 - np.sum(valid_mask) / len(pp_intervals))
        if abnormal_pct <= max_abnormal_percentage:
            segment['pp_intervals'] = clean_pp
            valid_segments.append(segment)
    #returns segments with valid peak to peak intervals
    return valid_segments

#Extract features from BVP and EDA signals 
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


if __name__ == "__main__":
    subjects_data = load_all_subjects()
    preprocessed_data = preprocess_subjects(subjects_data)
    all_segments, subject_durations = segmentation(preprocessed_data)
    all_labeled_segments = assign_labels(all_segments, subject_durations)
    valid_segments = extract_peak2peak(all_labeled_segments)
    features = extract_features(valid_segments)


