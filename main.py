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
cv_folds = 2
n_init = 1
random_state = 0
enable_reproducible_results(random_state)
results = {}
metrics = ["js", "wasserstein", "precision-recall", "authenticity", "domias"]

# ---------------------------------
# START BENCHMARKING

# load data
dataset = openml.datasets.get_dataset("Diabetes130US")
X, _, _, _ = dataset.get_data(dataset_format="dataframe")
X = X[:100]

# perform k fold CV
time_start = time.perf_counter()
for fold, (train, test) in enumerate(
    KFold(n_splits=cv_folds, shuffle=True, random_state=random_state).split(X)
):
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
            X_train.dataframe(), X_test.dataframe(), X_syn.dataframe(), metrics
        )
time_end = time.perf_counter()
results["timer"] = time_end - time_start
# final result dict is of form: {fold:{init:{metric:result}},timer:time}
# save results
if not os.path.exists("results"):
    os.makedirs("results")
with open(f"results/{generator}.json", "w") as f:
    json.dump(results, f, indent=4)
