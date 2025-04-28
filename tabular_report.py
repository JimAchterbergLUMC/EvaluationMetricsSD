# create an interpretable evaluation report for a single SD generator


from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins import Plugins
from synthcity.utils.reproducibility import enable_reproducible_results
import openml
from evaluation import report

# ---------------------------------
# BENCHMARK PARAMETERS

generator = "arf"
hparams = {}
# hparams = {
#     "n_iter": 300,
#     "generator_n_layers_hidden": 2,
#     "discriminator_n_layers_hidden": 2,
#     "generator_n_units_hidden": 256,
#     "discriminator_n_units_hidden": 256,
#     "lr": 2e-4,
#     "weight_decay": 1e-6,
#     "batch_size": 500,
#     "generator_dropout": 0,
#     "discriminator_dropout": 0.2,
#     "generator_nonlin": "relu",
#     "discriminator_nonlin": "leaky_relu",
#     "encoder_max_clusters": 10,
#     "clipping_value": 0,
#     "lambda_gradient_penalty": 10,
# }
seed = 0
enable_reproducible_results(seed)

# ---------------------------------
# START REPORTING

# load data
# dataset = openml.datasets.get_dataset(4541)
# X, _, _, _ = dataset.get_data(dataset_format="dataframe")
# X = X.drop(["encounter_id", "patient_nbr"], axis=1)
# X = X[:10000]

from sklearn.datasets import load_diabetes
import pandas as pd

X, y = load_diabetes(return_X_y=True, as_frame=True, scaled=False)
X = pd.concat([X, y], axis=1)


X = GenericDataLoader(data=X, random_state=seed, train_size=0.7)

hparams["random_state"] = seed
plugin = Plugins().get(generator, **hparams)

# unconditional generation (we do not consider a specific target feature)
plugin.fit(X.train())
X_syn = plugin.generate(len(X))

# create evaluation report (automatically saves as files in specified directory)
save_dir = "results/report"
# pass metrics with params
metrics = {
    # "domain_constraints": {"constraint_list": ["s1>=s2+s3"]},
    # "marginal_plots": {
    #     "discrete_features": ["sex"],
    #     "plot_cols": 4,
    #     "figsize": (10, 10),
    #     "single_fig": True,
    # },
    # "correlation_plots": {
    #     "discrete_features": ["sex"],
    #     "figsize": (10, 10),
    #     "single_fig": True,
    # },
    # "arm": {
    #     "discrete_features": ["sex"],
    #     "n_bins": 2,
    #     "min_support": 0.2,
    #     "min_confidence": 0.7,
    # },
    # "dwp": {
    #     "discrete_features": ["sex"],
    # },
    "projections": {
        "embedder": "pca",
        "plot": "marginal",
        "figsize": (10, 10),
        **{"n_components": 0.9},
    },
    # "nndr": {"plot": "boxplot"}
}
report(
    X.train().dataframe(),
    X.test().dataframe(),
    X_syn.dataframe(),
    save_dir=save_dir,
    **metrics
)
