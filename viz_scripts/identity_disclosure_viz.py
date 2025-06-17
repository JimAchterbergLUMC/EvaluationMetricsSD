import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(0)
n = 100
data = np.random.normal(0, 1, (n, 2))
syn = np.random.normal(0, 1, (n, 2))

outlier = np.array([[4, 3]])
outlier_syn = np.array([[3.9, 2.9]])

data = np.concatenate((data, outlier))
syn = np.concatenate((syn, outlier_syn))

cols = ["col_" + str(x) for x in list(range(data.shape[1]))]

data = pd.DataFrame(data, columns=cols)
syn = pd.DataFrame(syn, columns=cols)

data = data.assign(Dataset="Training Set")
syn = syn.assign(Dataset="Synthetic Data")

df = pd.concat((data, syn))

fig, ax = plt.subplots(figsize=(16, 6))

sns.scatterplot(
    df,
    x=cols[0],
    y=cols[1],
    hue="Dataset",
    s=150,
    alpha=0.7,
    palette={"Synthetic Data": "red", "Training Set": "blue"},
    ax=ax,
)


sns.move_legend(ax, loc="upper left", fontsize=16, title_fontsize=16)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticks([])
ax.set_yticks([])

sns.despine(fig, left=True)

plt.tight_layout()

plt.savefig("results/identity_disclosure_viz.png")
