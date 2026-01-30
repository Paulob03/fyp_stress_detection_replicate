from preprocess import *
import numpy as np


#Divide continuous BVP and EDA signals into 1 minnute segmans
#preprocessed_data = list of dicts with subject_id, bvp_filtered and eda_processed
def segmentation(preprocessed_data):
    segment_duration = 60
    bvp_fs=64.0
    eda_fs=64.0

    all_segments = []
    subject_durations = {} 


    for subject in preprocessed_data:
        bvp_signal = subject['bvp_filtered']
        eda_signal = subject['eda_processed']
        subject_id = subject['subject_id']

        bvp_samples_per_segment = int(segment_duration * bvp_fs)
        eda_samples_per_segment = int(segment_duration * eda_fs)

        bvp_segment_amount = len(bvp_signal) // bvp_samples_per_segment
        eda_segment_amount = len(eda_signal) // eda_samples_per_segment
        
 

 
        subject_durations[subject_id] = min(bvp_segment_amount, eda_segment_amount)
        num_segments = min(bvp_segment_amount, eda_segment_amount)

        for i in range(num_segments):
            bvp_start = i * bvp_samples_per_segment 
            bvp_end = bvp_start + bvp_samples_per_segment
            eda_start = i * eda_samples_per_segment
            eda_end = eda_start + eda_samples_per_segment

            all_segments.append({
                'subject_id': subject_id,
                'segment_idx': i,
                'bvp_segment': bvp_signal[bvp_start:bvp_end],
                'eda_segment': eda_signal[eda_start:eda_end]
            })

    #returns all segemnts: list of dict with segmented data
    return all_segments, subject_durations

#Labels segments as stress 1 or baseline 0 
def assign_labels(all_segments, subject_durations):
    for segment in all_segments:
        subject_id = segment['subject_id']
        segment_idx = segment['segment_idx']
        
        total_duration_min = subject_durations[subject_id]
        
        task4_end = total_duration_min - 5
        rest4_end = task4_end + 2
        task5_end = rest4_end + 1

        stress_periods = [
            (3, 13),    # Task 1
            (15, 20),   # Task 2
            (22, 25),   # Task 3
            (27, task4_end),        # Task 4
            (rest4_end, task5_end), # Task 5
        ]

        segment['label'] = 0

        for stress_start, stress_end in stress_periods:
            if stress_start <= segment_idx and segment_idx < stress_end:
                segment['label'] = 1
                break
    #Updated all segments with labels            
    return all_segments

def sliding_window_assign_labels(all_segments, subject_durations, window, step):
    window = window / 60 
    step_size = step / 60 
    for segment in all_segments:
        subject_id = segment['subject_id']
        segment_idx = segment['segment_idx']

        total_duration_min = subject_durations[subject_id]
        
        task4_end = total_duration_min - 4
        rest4_end = task4_end + 2
        task5_end = rest4_end + 1

        stress_periods = [
            (3, 13),    # Task 1
            (15, 20),   # Task 2
            (22, 25),   # Task 3
            (27, task4_end),        # Task 4
            (rest4_end, task5_end), # Task 5
        ]

        segment['label'] = 0

        for stress_start, stress_end in stress_periods:
            segment_start = segment_idx * step_size
            segment_end = segment_start + window
            if segment_start < stress_end and segment_end > stress_start:
                segment['label'] = 1
                break
    #Updated all segments with labels            
    return all_segments

def sliding_windows(preprocessed_data):
    segment_duration = 40 
    step_size = 30
    bvp_fs=64.0
    eda_fs=64.0

    all_segments = []
    subject_durations = {} 

    for subject in preprocessed_data:
        bvp_signal = subject['bvp_filtered']
        eda_signal = subject['eda_processed']
        subject_id = subject['subject_id']

        #The number of samples in one segment
        bvp_samples_per_segment = int(segment_duration * bvp_fs)
        eda_samples_per_segment = int(segment_duration * eda_fs)

        #The number of samples in a step forward
        bvp_segment_step = int(step_size * bvp_fs)
        eda_segment_step = int(step_size * eda_fs)

        #The number of windows per subject
        bvp_num_windows = (len(bvp_signal) - bvp_samples_per_segment) // bvp_segment_step + 1
        eda_num_windows = (len(eda_signal) - eda_samples_per_segment) // eda_segment_step + 1
        num_segments = min(bvp_num_windows, eda_num_windows) 

        #Number of minutes per subject
        protocol_length = len(bvp_signal) / (bvp_fs * 60)
 
        subject_durations[subject_id] = protocol_length
        
        #    
        task4_end = subject_durations[subject_id] - 5
        rest4_end = task4_end + 2
        task5_end = rest4_end + 1
        rest6_end = task5_end + 2
        change_periods = np.array([3, 13, 15, 20, 22, 25,  27, task4_end,rest4_end, task5_end, rest6_end])
                         #  [180, 780, 900, 1200, 1320, 1500, 1620] 
        change_periods *= 60

        seg_inx = 0
        seg_start = 0
        count_period = 0
        curr_label = 0 # 

        while seg_start + segment_duration <= change_periods[-1]:
            seg_end = seg_start + segment_duration


            # if the segment is between 2 different tasks            
            if seg_end > change_periods[count_period]: 
                seg_start = change_periods[count_period] # start of the next task
                count_period+=1
                curr_label = abs(curr_label-1)
                continue

            # compute stats for [seg_start, seg_end] with label curr-label
            else:
                seg_inx +=1
                print(f"Segment {seg_inx}: time {seg_start/60:.2f}-{seg_end/60:.2f} min, period {count_period}, label {curr_label}")
    
                bvp_start = int(seg_start * bvp_fs)
                bvp_end = int(seg_end * bvp_fs)
                bvp_start = int(seg_start * bvp_fs)
                bvp_end = int(seg_end * bvp_fs)
                eda_start = int(seg_start * eda_fs)
                eda_end = int(seg_end * eda_fs)
                all_segments.append({
                    'subject_id': subject_id,
                    'segment_idx': seg_inx,
                    'bvp_segment': bvp_signal[bvp_start:bvp_end],
                    'eda_segment': eda_signal[eda_start:eda_end],
                    'label': curr_label

                })
                seg_start += step_size


                
            

         

     
        '''for i in range(num_segments):
            bvp_start = i * bvp_segment_step
            bvp_end = bvp_start + bvp_samples_per_segment  
            eda_start = i * eda_segment_step
            eda_end = eda_start + eda_samples_per_segment 

            all_segments.append({
                'subject_id': subject_id,
                'segment_idx': i,
                'bvp_segment': bvp_signal[bvp_start:bvp_end],
                'eda_segment': eda_signal[eda_start:eda_end]
             })'''

    #returns all segemnts: list of dict with segmented data
    return all_segments


if __name__ == "__main__":
 
    subjects_data = load_all_subjects()
    preprocessed_data = preprocess_subjects(subjects_data)
    #all_segments, subject_durations = segmentation(preprocessed_data)
    #all_labeled_segments = assign_labels(all_segments, subject_durations)
    all_segments = sliding_windows(preprocessed_data)




    