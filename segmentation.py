from preprocess import *


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
            if stress_start <= segment_idx and segment_idx < stress_end:
                segment['label'] = 1
                break
    #Updated all segments with labels            
    return all_segments



if __name__ == "__main__":
 
    subjects_data = load_all_subjects()
    preprocessed_data = preprocess_subjects(subjects_data)
    all_segments, subject_durations = segmentation(preprocessed_data)
    all_labeled_segments = assign_labels(all_segments, subject_durations)

    print(all_labeled_segments[0])



    