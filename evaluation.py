# computes metrics to construct a benchmark or evaluation report

import pandas as pd
import os
import shutil

from utils import preprocess_eval, determine_feature_types

from metrics.fidelity import (
    DomainConstraint,
    AssociationRuleMining,
    DWP,
    FeatureWisePlots,
    ClassifierTest,
    ClusteringTest,
    CorrelationPlots,
    Projections,
    PRDC,
    Wasserstein,
    MMD,
)

from metrics.privacy import DOMIAS, Authenticity

METRICS = {
    "domain_constraints": DomainConstraint,
    "arm": AssociationRuleMining,
    "dwp": DWP,
    "featurewise_plots": FeatureWisePlots,
    "classifier_test": ClassifierTest,
    "clustering_test": ClusteringTest,
    "correlation_plots": CorrelationPlots,
    "projections": Projections,
    "prdc": PRDC,
    "wasserstein": Wasserstein,
    "mmd": MMD,
    "authenticity": Authenticity,
    "domias": DOMIAS,
}


def benchmark(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_syn: pd.DataFrame,
    metrics: dict,
):
    """
    Evaluates a bunch of metrics for a synthetic dataset w.r.t. a real test set.
    Returns a dictionary like {metric_name:result}.

    metrics: dict of dicts. Each metric has a dict containing parameters relevant to initializing that metric.
    """

    # one hot, label encode, minmax scale
    # TBD: separate metric computation for those which do not require preprocessing
    # -> no preprocessing: featurewise plots, correlations, dwp, domain constraint, utility metrics, projections maybe? classifier test (due to skf),
    X_tr_scaled, X_te_scaled, X_syn_scaled = preprocess_eval(
        X_train, X_test, X_syn, ohe_threshold=15
    )

    dict_ = {}
    for metric_ in metrics.keys():
        metric = METRICS[metric_](**metrics[metric_])
        if "domias" in metric_.lower():
            # only domias requires training data
            metric_result = metric.evaluate(X_tr_scaled, X_te_scaled, X_syn_scaled)
        else:
            metric_result = metric.evaluate(
                X_te_scaled,
                X_syn_scaled,
            )

        if type(metric_result) == dict:
            dict_.update(metric_result)
        else:
            try:
                dict_[metric_] = metric_result
            except:
                # don't add if not possible (perhaps a metric which doesnt output anything)
                pass

    return dict_


# def report(
#     train: pd.DataFrame,
#     test: pd.DataFrame,
#     syn: pd.DataFrame,
#     save_dir: str,
#     random_state: int,
#     **report_params,
# ):
#     if os.path.exists(save_dir):
#         shutil.rmtree(save_dir)
#     os.makedirs(save_dir, exist_ok=True)

#     # compute evals and save as files in save_dir
#     # -------------------------------------------------------
#     # fidelity

#     constraints(
#         test,
#         syn,
#         f"{save_dir}/constraints",
#         constraint_list=report_params["constraints"],
#     )

#     marginal_plots(test, syn, f"{save_dir}/marginal_plots")

#     correlations(test, syn, f"{save_dir}/correlations")

#     # Association Rule Mining

#     # Dimension Wise Prediction

#     # projections
#     project(
#         test, syn, f"{save_dir}/projections", random_state, type_list=["pca", "tsne"]
#     )

#     # -------------------------------------------------------
#     # privacy

#     # MIA: make assumption regarding
