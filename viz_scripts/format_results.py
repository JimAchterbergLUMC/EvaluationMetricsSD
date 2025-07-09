# formats results from JSON files to readable tables with averages, std, etc.

import json
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os
import re

# open json file
generator = "arf"
with open(f"results/{generator}.json", "r") as f:
    data = json.load(f)


def print_table(data):

    # create df
    fold_data = []
    for fold, metrics in data.items():
        for init, values in metrics.items():
            # Extract fold and init values
            fold_value = int(fold.split(":")[1].strip())
            init_value = int(init.split(":")[1].strip())
            values["fold"] = fold_value
            values["init"] = init_value
            fold_data.append(values)
    df = pd.DataFrame(fold_data)

    print(df.mean().round(3))  # type: ignore

    print(df.std().round(3))  # type: ignore


print_table(data)
