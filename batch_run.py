
from load_data import *
from preprocess import *
from segmentation import *
from feature_extraction import *
from model import *
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict

CONFIGS = [
    {"window":  15, "step":  5},
]

RESULTS_FILE = "window_sweep_results.csv"
CM_DIR       = "confusion_matrices"
os.makedirs(CM_DIR, exist_ok=True)

print("Starting")
subjects_data     = load_all_subjects()
preprocessed_data = preprocess_subjects(subjects_data)

results = []

for cfg in CONFIGS:
    w, s  = cfg["window"], cfg["step"]
    label = f"win{w}_step{s}"
    print(f"{'='*60}")
    print(f"Running: {label}  ({datetime.now().strftime('%H:%M:%S')})")

    try:
        all_segments   = sliding_windows(preprocessed_data, segment_duration=w, step_size=s)
        valid_segments = extract_peak2peak(all_segments)
        features       = extract_features(valid_segments)

        n_segments = len(features)
        n_stress   = sum(1 for f in features if f['label'] == 1)
        n_rest     = n_segments - n_stress
        print(f"  Segments: {n_segments}  (stress={n_stress}, rest={n_rest})")

        cv, best_estimator, X, y, groups, gkf = model(features)

        # Confusion matrix
        y_pred = cross_val_predict(best_estimator, X, y, groups=groups, cv=gkf)
        cm     = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['REST', 'STRESS'])
        fig, ax = plt.subplots(figsize=(6, 5))
        cm_display.plot(ax=ax)
        ax.set_title(f"{label}\nbal_acc={np.mean(cv['test_balanced_accuracy']):.3f}  f1={np.mean(cv['test_f1']):.3f}")
        cm_path = os.path.join(CM_DIR, f"cm_{label}.png")
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Confusion matrix saved to {cm_path}")

        row = {
            "label":       label,
            "window_s":    w,
            "step_s":      s,
            "n_segments":  n_segments,
            "n_stress":    n_stress,
            "n_rest":      n_rest,
            "train_acc":   round(np.mean(cv['train_accuracy']), 4),
            "test_acc":    round(np.mean(cv['test_accuracy']), 4),
            "bal_acc":     round(np.mean(cv['test_balanced_accuracy']), 4),
            "f1":          round(np.mean(cv['test_f1']), 4),
            "precision":   round(np.mean(cv['test_precision']), 4),
            "recall":      round(np.mean(cv['test_recall']), 4),
            "overfit_gap": round(np.mean(cv['train_accuracy']) - np.mean(cv['test_accuracy']), 4),
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "status":      "ok"
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        row = {"label": label, "window_s": w, "step_s": s, "status": f"ERROR: {e}"}

    results.append(row)
    pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)
    print(f"  Saved to {RESULTS_FILE}")


print("Finished")
print(pd.DataFrame(results)[["label", "test_acc", "bal_acc", "f1", "overfit_gap", "status"]].to_string())