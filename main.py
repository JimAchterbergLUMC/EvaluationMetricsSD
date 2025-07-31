import json
import os
import time

from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.utils.reproducibility import clear_cache
from synthcity.plugins import Plugins
from synthcity.utils.reproducibility import enable_reproducible_results

# import openml
from sklearn.model_selection import KFold
from utils.utils import clear_dir
from evaluation import evaluate
from sklearn.model_selection import train_test_split

# from sklearn.datasets import load_diabetes
import pandas as pd

# start with clean workspace
clear_dir("workspace")

cv_folds = 3
n_init = 3
seed = 0
enable_reproducible_results(seed)

# ---------------------------------
# BENCHMARK PARAMETERS
hparams_all = {
    "arf": {},
    "bayesian_network": {"struct_max_indegree": 2},
    "ctgan": {
        "n_iter": 300,
        "generator_n_layers_hidden": 2,
        "discriminator_n_layers_hidden": 2,
        "generator_n_units_hidden": 256,
        "discriminator_n_units_hidden": 256,
        "lr": 2e-4,
        "weight_decay": 1e-6,
        "batch_size": 500,
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
        "encoder_n_layers_hidden": 2,
        "decoder_n_layers_hidden": 2,
        "encoder_n_units_hidden": 128,
        "decoder_n_units_hidden": 128,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 500,
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
        "batch_size": 500,
        "num_timesteps": 200,
        "gaussian_loss_type": "mse",
        "scheduler": "linear",
        "dim_embed": 128,
        "model_type": "mlp",
        "model_params": dict(n_layers_hidden=2, n_units_hidden=128, dropout=0.0),
    },
}

metrics = {
    "wasserstein": {},
    "prdc": {},
    "classifier_test": {"random_state": seed},
    "authenticity": {},
    "domias": {
        "reduction": "pca",
        "n_components": 0.99,
        "random_state": seed,
        "metric": "roc_auc",
        "ref_prop": 0.5,  # reference proportion taken from test set
        "member_prop": 1,  # member proportion taken from training set
        "quasi_identifiers": [],
        "predict_top": 0.05,
    },
    # "domain_constraints": {"constraint_list": ["bp_systolic>bp_diastolic"]},
    # "featurewise_plots": {"figsize": (10, 10)},
    # "correlation_matrices": {"figsize": (10, 5)},
    # "association_rules": {
    #     "n_bins": 3,
    #     "min_support": 0.5,
    #     "min_confidence": 0.75,
    # },
    # "aia": {
    #     "quasi_identifiers": ["age", "sex", "bmi"],
    #     "sensitive_attributes": ["los", "admission_location"],
    # },
    # "nndr": {},
}

generator = "ddpm"
# for generator in hparams_all.keys():
hparams = hparams_all[generator]
results = {}

# ---------------------------------
# START BENCHMARKING

# load data
X = pd.read_csv("data/cohort.csv")
discrete_features = [
    "sex",
    "mortality",
    "ethnicity",
    "marital_status",
    "admission_type",
    "admission_location",
]

if cv_folds == 1 and n_init == 1:
    # No CV, single run
    print("Single run: no cross-validation.")
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=seed)
    X_train = GenericDataLoader(data=X_train)
    X_test = GenericDataLoader(data=X_test)

    hparams["random_state"] = seed
    plugin = Plugins().get(generator, **hparams)

    # Time the training phase
    start_time = time.time()
    plugin.fit(X_train)
    training_time = time.time() - start_time

    X_syn = plugin.generate(len(X))

    results["report"] = evaluate(
        X_train.dataframe(),
        X_test.dataframe(),
        X_syn.dataframe(),
        metrics,
        discrete_features=discrete_features,
    )

    # Add timing information to results
    results["training.time.minutes"] = training_time / 60
else:
    # perform k fold CV
    for fold, (train, test) in enumerate(
        KFold(n_splits=cv_folds, shuffle=True, random_state=seed).split(X)
    ):
        print(f"fold: {fold}")
        results[f"fold: {fold}"] = {}
        X_train = GenericDataLoader(data=X.iloc[train])
        X_test = GenericDataLoader(data=X.iloc[test])
        for i in range(n_init):
            print(f"init: {i}")
            hparams["random_state"] = i
            clear_cache()
            plugin = Plugins().get(generator, **hparams)

            # Time the training phase
            start_time = time.time()
            plugin.fit(X_train)
            training_time = time.time() - start_time

            X_syn = plugin.generate(len(X))

            clear_cache()
            clear_dir("workspace")
            results[f"fold: {fold}"][f"init: {i}"] = evaluate(
                X_train.dataframe(),
                X_test.dataframe(),
                X_syn.dataframe(),
                metrics,
                discrete_features=discrete_features,
            )

            # Add timing information to results
            results[f"fold: {fold}"][f"init: {i}"]["training.time.minutes"] = (
                training_time / 60
            )


# save results
os.makedirs("results", exist_ok=True)
with open(f"results/{generator}.json", "w") as f:
    json.dump(results, f, indent=4)

clear_cache()
clear_dir("workspace")

# benchmarking results can be pretty printed using viz_scripts/format_results.py
