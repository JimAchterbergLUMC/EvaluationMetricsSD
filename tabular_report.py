import os
import json
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins import Plugins
from synthcity.utils.reproducibility import enable_reproducible_results
import openml
from evaluation import evaluate
from sklearn.datasets import load_diabetes
import pandas as pd

# ---------------------------------
# BENCHMARK PARAMETERS

generator = "ddpm"
hparams = {
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
}
seed = 0
enable_reproducible_results(seed)

# ---------------------------------
# START REPORTING

# load data
# import numpy as np

# dataset = openml.datasets.get_dataset(4541)
# X, _, _, _ = dataset.get_data(dataset_format="dataframe")
# X = X.drop(["encounter_id", "patient_nbr"], axis=1)
# X = X[:500]
# discrete_features = np.array(dataset.get_features_by_type("nominal")) - 2
# discrete_features = X.columns[discrete_features].tolist()

X, y = load_diabetes(return_X_y=True, as_frame=True, scaled=False)
X = pd.concat([X, y], axis=1)
X["sex"] = X["sex"].map({1: "female", 2: "male"})
discrete_features = ["sex"]

X = GenericDataLoader(data=X, random_state=seed, train_size=0.5)

hparams["random_state"] = seed
plugin = Plugins().get(generator, **hparams)

# unconditional generation (we do not consider a specific target feature)
plugin.fit(X.train())
X_syn = plugin.generate(len(X))

# pass metrics with params
metrics = {
    "domain_constraints": {"constraint_list": ["s1>=s2+s3"]},
    "featurewise_plots": {"figsize": (10, 10)},
    "correlation_matrices": {"figsize": (10, 5)},
    "association_rules": {
        "n_bins": 3,
        "min_support": 0.3,
        "min_confidence": 0.75,
    },
    "dwp": {},
    "authenticity": {},
    "aia": {
        "quasi_identifiers": ["age", "sex", "bmi", "bp"],
        "sensitive_attributes": ["s1", "s6"],
    },
    # "projections-pca": {"embedder": "pca", "figsize": (10, 10), "n_components": 0.8},
    "domias-pr": {
        "reduction": "pca",
        "n_components": 0.99,
        "random_state": seed,
        "metric": "precision-recall",
        "ref_prop": 0.5,  # reference proportion taken from test set
        "member_prop": 1,  # member proportion taken from training set
        "quasi_identifiers": [],
        "predict_top": 0.05,
    },
    "domias-roc": {
        "reduction": "pca",
        "n_components": 0.99,
        "random_state": seed,
        "metric": "roc_auc",
        "ref_prop": 0.5,  # reference proportion taken from test set
        "member_prop": 1,  # member proportion taken from training set
        "quasi_identifiers": [],
    },
    # "projections-umap": {"embedder": "umap", "markersize": 100},
    "nndr": {},
    # "mia": {
    #     "generator": generator,
    #     "generator_hparams": hparams,
    #     "random_state": seed,
    #     "metric": "f1",
    # },
}

results = evaluate(
    X.train().dataframe(),
    X.test().dataframe(),
    X_syn.dataframe(),
    metrics,
    discrete_features=discrete_features,
)
os.makedirs("results", exist_ok=True)
with open(f"results/{generator}.json", "w") as f:
    json.dump(results, f, indent=4)
