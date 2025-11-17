import pandas as pd
import numpy as np
import os

DATA_DIR = "Subjects"


def load_subject_signals(subject_folder):

    bvp_df = pd.read_csv(os.path.join(subject_folder, 'BVP.csv'), header=None)
    bvp_signal = bvp_df.iloc[:, 0].astype(float).values
    eda_df = pd.read_csv(os.path.join(subject_folder, 'EDA.csv'), header=None)
    eda_df[0] = eda_df[0].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    eda_signal = eda_df.iloc[:, 0].astype(float).values

    return bvp_signal, eda_signal

def load_all_subjects():
    DATA_DIR = "Subjects"
    all_subjects = []
    for subject_num in range(1, 30):
        subject_id = f"subject_{subject_num:02d}"
        folder = os.path.join(DATA_DIR, subject_id)
        
        bvp, eda = load_subject_signals(folder)
        all_subjects.append({
            'subject_id': subject_id,
            'bvp_signal': bvp,
            'bvp_fs': 64.0,
            'eda_signal': eda,
            'eda_fs': 4.0
        })
    
    return all_subjects 


if __name__ == "__main__":

    all_subjects = load_all_subjects()
    
