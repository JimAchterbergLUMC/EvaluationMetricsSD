import time
import json
import os

from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins import Plugins
from synthcity.utils.reproducibility import enable_reproducible_results
import openml
from sklearn.model_selection import KFold

from evaluation import benchmark

# ---------------------------------
# BENCHMARK PARAMETERS

generator = "arf"
hparams = {}
cv_folds = 2
n_init = 1
seed = 0
enable_reproducible_results(seed)
results = {}
metrics = {"wasserstein": {}, "prdc": {}, "mmd": {}, "authenticity": {}, "domias": {}}
# ---------------------------------
# START BENCHMARKING

# load data
# dataset = openml.datasets.get_dataset(4541)
# X, _, _, _ = dataset.get_data(dataset_format="dataframe")
# X = X.drop(["encounter_id", "patient_nbr"], axis=1)
# X = X[:100]

from sklearn.datasets import load_diabetes
import pandas as pd

X, y = load_diabetes(as_frame=True, return_X_y=True, scaled=False)
X = pd.concat([X, y], axis=1)

# perform k fold CV
time_start = time.perf_counter()
for fold, (train, test) in enumerate(
    KFold(n_splits=cv_folds, shuffle=True, random_state=seed).split(X)
):
    print(f"fold: {fold}")
    results[f"fold: {fold}"] = {}
    # get train-test data
    X_train = GenericDataLoader(data=X.iloc[train])
    X_test = GenericDataLoader(data=X.iloc[test])

    # synthesize for multiple initializations
    for i in range(n_init):
        print(f"init: {i}")
        hparams["random_state"] = i
        plugin = Plugins().get(generator, **hparams)
        # unconditional generation (we do not consider a specific target feature)
        plugin.fit(X_train)
        X_syn = plugin.generate(len(test))
        # evaluation
        results[f"fold: {fold}"][f"init: {i}"] = benchmark(
            X_train.dataframe(),
            X_test.dataframe(),
            X_syn.dataframe(),
            metrics,  # we use the same random state for metrics across initializations
        )
time_end = time.perf_counter()
results["timer"] = time_end - time_start

# save results
os.makedirs("results/benchmark", exist_ok=True)
with open(f"results/benchmark/{generator}.json", "w") as f:
    json.dump(results, f, indent=4)
