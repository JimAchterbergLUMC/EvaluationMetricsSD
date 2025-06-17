import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# random data
np.random.seed(0)
n = 20
train = np.random.normal(0, 1, (n, 2))
test = np.random.normal(0, 1, (n, 2))
syn = train + 5e-2
cols = ["col_" + str(x) for x in list(range(train.shape[1]))]
train = pd.DataFrame(train, columns=cols)
syn = pd.DataFrame(syn, columns=cols)
train = train.assign(Dataset="Training Set")
syn = syn.assign(Dataset="Synthetic Data")
test = pd.DataFrame(test, columns=cols)
test = test.assign(Dataset="Test Set")

data = pd.concat((train, test, syn))

attack = pd.concat((train[: int(n / 2)], test[: int(n / 2)]))
attack["Dataset"] = "Attack Set"
attack = pd.concat((attack, syn))

fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

sns.scatterplot(
    data,
    x=cols[0],
    y=cols[1],
    hue="Dataset",
    palette={"Synthetic Data": "red", "Training Set": "blue", "Test Set": "green"},
    s=150,
    alpha=0.7,
    ax=axs[0],
)

sns.scatterplot(
    attack,
    x=cols[0],
    y=cols[1],
    hue="Dataset",
    s=150,
    palette={"Synthetic Data": "red", "Attack Set": "black"},
    alpha=0.7,
    ax=axs[1],
)

for ax in axs:
    sns.move_legend(ax, loc="upper left", fontsize=16, title_fontsize=16)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

sns.despine(fig, left=True)

plt.tight_layout()

plt.savefig("results/membership_disclosure_viz.png")
