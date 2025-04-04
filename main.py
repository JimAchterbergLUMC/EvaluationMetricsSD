import time
import json
import os

from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins import Plugins
from synthcity.utils.reproducibility import enable_reproducible_results
import openml
from sklearn.model_selection import KFold

from metrics import evaluate

# ---------------------------------
# BENCHMARK PARAMETERS

generator = "marginal_distributions"
hparams = {}
cv_folds = 10
n_init = 1
seed = 0
enable_reproducible_results(seed)
results = {}
metrics = [
    "mmd",
    "wasserstein",
    # "precision-recall", "authenticity", "domias"
]

# ---------------------------------
# START BENCHMARKING

# load data
dataset = openml.datasets.get_dataset("Diabetes130US")
X, _, _, _ = dataset.get_data(dataset_format="dataframe")
X = X.drop(["encounter_id", "patient_nbr"], axis=1)

# perform k fold CV
time_start = time.perf_counter()
for fold, (train, test) in enumerate(
    KFold(n_splits=cv_folds, shuffle=True, random_state=seed).split(X)
):
    print(f"fold: {fold}")
    results[fold] = {}
    # get train-test data
    X_train = GenericDataLoader(data=X.iloc[train])
    X_test = GenericDataLoader(data=X.iloc[test])

    # synthesize for multiple initializations
    for i in range(n_init):
        hparams["random_state"] = i
        plugin = Plugins().get(generator, **hparams)
        # unconditional generation (we do not consider a specific target feature)
        plugin.fit(X_train)
        X_syn = plugin.generate(len(test))
        # evaluation
        results[fold][i] = evaluate(
            X_train.dataframe(),
            X_test.dataframe(),
            X_syn.dataframe(),
            metrics,
            random_state=seed,  # we use the same random state for metrics across initializations
        )
time_end = time.perf_counter()
results["timer"] = time_end - time_start
# save results
if not os.path.exists("results"):
    os.makedirs("results")
with open(f"results/{generator}.json", "w") as f:
    json.dump(results, f, indent=4)
