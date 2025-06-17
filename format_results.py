# formats results from JSON files to readable tables with averages, std, etc.

import json
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os
import re

# open json file
generator = "ddpm"
with open(f"results/{generator}.json", "r") as f:
    data = json.load(f)


def print_table(data):

    # create df
    fold_data = []
    for fold, metrics in data.items():
        if fold != "timer":
            for init, values in metrics.items():
                # Extract fold and init values
                fold_value = int(fold.split(":")[1].strip())
                init_value = int(init.split(":")[1].strip())
                values["fold"] = fold_value
                values["init"] = init_value
                fold_data.append(values)
    df = pd.DataFrame(fold_data)
    df["timer"] = data["timer"]

    print(df.mean().round(3))

    print(df.std().round(3))


def parse_dwp(data):

    data = {k: v for k, v in data.items() if "DWP" in k}
    # parse entries
    rows = []
    for key, value in data.items():
        # match the pattern
        match = re.match(r"^DWP (\w+) (\w+) \((.*), (\d+) folds, (.*)\)$", key)
        if match:
            data_type, feature, model, folds, metric = match.groups()
            rows.append(
                {
                    "type": data_type,
                    "feature": feature,
                    "model": model,
                    "folds": int(folds),
                    "metric": metric,
                    "value": value,
                }
            )

    # convert to DataFrame
    df = pd.DataFrame(rows)
    return df


def plot_dwp(df, path="results/dwp/", figsize=(10, 5)):

    df["type"] = df["type"].replace("SD", "Synthetic")
    df["type"] = df["type"].replace("RD", "Real")

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=df,
        x="feature",
        y="value",
        hue="type",
        palette={"Real": "blue", "Synthetic": "red"},
        alpha=0.5,
        ax=ax,
    )
    sns.despine()
    os.makedirs(path, exist_ok=True)
    ax.set_ylabel("Prediction Score")
    ax.set_xlabel("Target Feature")
    plt.savefig(f"{path}/dwp.pdf")


df = parse_dwp(data)
print(df)
# plot_dwp(df)
