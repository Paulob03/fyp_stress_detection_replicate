import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

with_feats    = pd.read_csv("window_sweep_results.csv")
without_feats = pd.read_csv("window_sweep_no_new_features.csv")

labels = with_feats["label"].str.replace("win", "").str.replace("_step", "/") + "s"
x      = np.arange(len(labels))
w      = 0.35

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("With vs Without New Features", fontsize=14, fontweight="bold", y=1.02)

metrics = [
    ("bal_acc",     "Balanced Accuracy", (0.68, 0.95)),
    ("test_acc", "Accuracy",          (0.68, 0.95)),
    ("f1", "F1 Score",       (0,  0.95)),
]

for ax, (col, title, ylim) in zip(axes, metrics):
    b1 = ax.bar(x - w/2, with_feats[col],    w, label="With new features",    color="#6366f1", alpha=0.9)
    b2 = ax.bar(x + w/2, without_feats[col], w, label="Without new features", color="#f59e0b", alpha=0.9)

    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(ylim)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
        # Annotate difference
    for i, (v1, v2) in enumerate(zip(with_feats[col], without_feats[col])):
        diff = v1 - v2
        color = "#16a34a" if (diff < 0 and col == "accuracy") or (diff > 0 and col != "accuracy") else "#dc2626"
        sign  = "+" if diff > 0 else ""
        ax.text(i, max(v1, v2) + 0.003, f"{sign}{diff*100:.1f}%", ha="center", fontsize=7, color=color, fontweight="bold")

plt.tight_layout()
plt.savefig("feature_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to feature_comparison.png")