# formats results from JSON files to readable tables with averages, std, etc.

import json
import pandas as pd

# open json file
generator = "arf"
with open(f"results/{generator}.json", "r") as f:
    data = json.load(f)

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
print(df)

per_fold = df.groupby("fold").mean()
print(per_fold)

avg = per_fold.mean()
std = per_fold.std()

print(avg)
print(std)
