# create an interpretable evaluation report for a single (best) SD generator


from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins import Plugins
from synthcity.utils.reproducibility import enable_reproducible_results
import openml
from metrics import report

# ---------------------------------
# BENCHMARK PARAMETERS

generator = "arf"
hparams = {}
seed = 0
enable_reproducible_results(seed)

# ---------------------------------
# START REPORTING

# load data
# dataset = openml.datasets.get_dataset("Diabetes130US")
# X, _, _, _ = dataset.get_data(dataset_format="dataframe")
# X = X.drop(["encounter_id", "patient_nbr"], axis=1)
# X = X[:100]


from sklearn.datasets import load_diabetes
import pandas as pd

X, y = load_diabetes(as_frame=True, return_X_y=True, scaled=False)
X = pd.concat([X, y], axis=1)

X = GenericDataLoader(data=X, random_state=seed, train_size=0.8)

hparams["random_state"] = seed
plugin = Plugins().get(generator, **hparams)

# unconditional generation (we do not consider a specific target feature)
plugin.fit(X.train())
X_syn = plugin.generate(len(X.test()))

# create evaluation report (automatically saves as files in specified directory)
save_dir = "results/report"
report(X.train().dataframe(), X.test().dataframe(), X_syn.dataframe(), save_dir, seed)
