import time
import json
import os

from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.utils.reproducibility import clear_cache
from synthcity.plugins import Plugins
from synthcity.utils.reproducibility import enable_reproducible_results
import openml
from sklearn.model_selection import KFold
from utils import clear_dir
from evaluation import evaluate

from sklearn.datasets import load_diabetes
import pandas as pd


# start with clean workspace
clear_dir("workspace")

# ---------------------------------
# BENCHMARK PARAMETERS
hparams_all = {
    "arf": {},
    "bayesian_network": {"struct_max_indegree": 2},
    "ctgan": {
        "n_iter": 300,
        "generator_n_layers_hidden": 1,
        "discriminator_n_layers_hidden": 1,
        "generator_n_units_hidden": 128,
        "discriminator_n_units_hidden": 128,
        "lr": 2e-4,
        "weight_decay": 1e-6,
        "batch_size": 32,
        "generator_dropout": 0,
        "discriminator_dropout": 0.2,
        "generator_nonlin": "relu",
        "discriminator_nonlin": "leaky_relu",
        "encoder_max_clusters": 10,
        "clipping_value": 0,
        "lambda_gradient_penalty": 10,
    },
    "tvae": {
        "n_iter": 300,
        "n_units_embedding": 128,
        "encoder_n_layers_hidden": 1,
        "decoder_n_layers_hidden": 1,
        "encoder_n_units_hidden": 128,
        "decoder_n_units_hidden": 128,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 32,
        "encoder_dropout": 0,
        "decoder_dropout": 0,
        "encoder_nonlin": "relu",
        "decoder_nonlin": "relu",
        "loss_factor": 2,
        "data_encoder_max_clusters": 10,
        "clipping_value": 0,
    },
    "ddpm": {
        "n_iter": 1000,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 64,
        "num_timesteps": 100,
        "gaussian_loss_type": "mse",
        "scheduler": "linear",
        "dim_embed": 128,
        "model_type": "mlp",
        "model_params": dict(n_layers_hidden=2, n_units_hidden=128, dropout=0.0),
    },
}

# for generator in ["arf", "bayesian_network", "ctgan", "tvae", "ddpm"]:

generator = "ddpm"
hparams = hparams_all[generator]
cv_folds = 3
n_init = 3
seed = 0
enable_reproducible_results(seed)
results = {}
# we can add the same metric multiple times with different parameters by adding a dash to the metric name
metrics = {
    "wasserstein": {},
    "prdc": {},
    "authenticity": {},
    "domias": {
        "reduction": "pca",
        "n_components": 0.99,
        "random_state": seed,
        "metric": "roc_auc",
        "ref_prop": 0.5,  # reference proportion taken from test set
        "member_prop": 1,  # member proportion taken from training set
        "quasi_identifiers": [],
    },
    "classifier_test": {"random_state": seed},
}
# ---------------------------------
# START BENCHMARKING

# load data
# dataset = openml.datasets.get_dataset(4541)
# X, _, _, _ = dataset.get_data(dataset_format="dataframe")
# X = X.drop(["encounter_id", "patient_nbr"], axis=1)
# X = X[:100]

X, y = load_diabetes(as_frame=True, return_X_y=True, scaled=False)
X = pd.concat([X, y], axis=1)
X["sex"] = X["sex"].map({1: "female", 2: "male"})
discrete_features = ["sex"]

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
        # clear GPU cache before SD generation
        clear_cache()
        plugin = Plugins().get(generator, **hparams)
        # unconditional generation (we do not consider a specific target feature)
        plugin.fit(X_train)
        X_syn = plugin.generate(len(X))
        # clear GPU cache after SD generation (some eval metrics also require GPU memory)
        clear_cache()
        # clear workspace cache
        clear_dir("workspace")
        # evaluation
        results[f"fold: {fold}"][f"init: {i}"] = evaluate(
            X_train.dataframe(),
            X_test.dataframe(),
            X_syn.dataframe(),
            metrics,
            discrete_features=discrete_features,
        )
time_end = time.perf_counter()
results["timer"] = time_end - time_start

# save results
os.makedirs("results", exist_ok=True)
with open(f"results/{generator}.json", "w") as f:
    json.dump(results, f, indent=4)
