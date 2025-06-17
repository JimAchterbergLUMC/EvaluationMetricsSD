import seaborn as sns
from matplotlib import pyplot as plt
import os

import numpy as np
import pandas as pd
from evaluation import evaluate
from sklearn.model_selection import train_test_split

# simulate data
seed = 0
np.random.seed(seed)


Nr = 1000
Ns = 1000
data = pd.DataFrame(["Real"] * Nr + ["Synthetic"] * Ns, columns=["Source"])
real = np.random.normal(0, 1, (Nr))
syn = np.random.normal(1, 1, (Ns))
data["Shift"] = np.concatenate((real, syn))
syn = np.random.normal(0, 1, (int((2 / 3) * Ns)))
syn = np.concatenate((syn, np.random.normal(4.25, 1, (Ns - len(syn)))))
data["Mode Invention"] = np.concatenate((real, syn))
syn = np.random.normal(0, 1, (Ns))
real = np.random.normal(0, 1, (int((2 / 3) * Nr)))
real = np.concatenate((real, np.random.normal(4.5, 1, (Nr - len(real)))))
data["Mode Collapse"] = np.concatenate((real, syn))
real = np.random.normal(0, 1, (Nr))
syn = np.random.normal(0, 2.25, (Ns))
data["Underfit"] = np.concatenate((real, syn))
real = np.random.normal(0, 2.35, (Nr))
syn = np.random.normal(0, 1, (Ns))
data["Overfit"] = np.concatenate((real, syn))


datasets = ["Shift", "Mode Invention", "Mode Collapse", "Underfit", "Overfit"]

fig, axs = plt.subplots(2, 3, figsize=(15, 5))  # 2 rows, 3 columns

# Flatten the axes array for easier indexing
axs = axs.flatten()

# Iterate over datasets and their respective axes
for i, col in enumerate(datasets):
    ax = axs[i]
    Xreal = data[data["Source"] == "Real"]
    Xsyn = data[data["Source"] == "Synthetic"]

    # do a train-test split
    X_train, X_test = train_test_split(
        Xreal[[col]].reset_index(drop=True), test_size=0.999, random_state=seed
    )

    # evaluate the required metrics
    metric_results = evaluate(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        X_syn=Xsyn[[col]].reset_index(drop=True),
        metrics={
            "wasserstein": {},
            "prdc": {},
            "classifier_test": {"random_state": seed},
            "jensenshannon": {"random_state": seed, "embed": None},
        },
        discrete_features=[],
    )

    print(f"DATASET: {col} \n {metric_results}")

    sns.kdeplot(
        data=data,
        x=col,
        hue="Source",
        palette={"Real": "blue", "Synthetic": "red"},
        ax=ax,
        fill=True,
        alpha=0.3,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{col}", fontsize=12)


# Place the legend in the location of the empty subplot (third column of the first row)
handles, labels = axs[0].legend_.legend_handles, [
    t.get_text() for t in axs[0].legend_.texts
]
fig.legend(
    handles,
    labels,
    # loc="best",
    title="Dataset",
    title_fontsize=16,
    fontsize=16,
    markerscale=2,
    bbox_to_anchor=(0.9, 0.4),
)

# Remove sub-legends
for ax in axs[: len(datasets)]:
    ax.get_legend().remove()

# Hide the last empty subplot
axs[5].set_axis_off()

plt.tight_layout()  # rect=[0, 0, 0.85, 1]

sns.despine(fig=fig, left=True, bottom=True)

save_dir = "results/failure_modes"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(f"{save_dir}/fig.png")
