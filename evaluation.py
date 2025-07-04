# TBD: remove hardcoding metrics into which type of data is accepted and whether discrete features need to be added.
# this can be done more OO, e.g., through class properties.

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
    CorrelationMatrices,
    Projections,
    PRDC,
    Wasserstein,
    MMD,
    JensenShannon,
)

from metrics.privacy import DOMIAS, Authenticity, NNDR, MIA, AttributeInferenceAttack

METRICS = {
    "domain_constraints": DomainConstraint,
    "association_rules": AssociationRuleMining,
    "dwp": DWP,
    "featurewise_plots": FeatureWisePlots,
    "classifier_test": ClassifierTest,
    "correlation_matrices": CorrelationMatrices,
    "projections": Projections,
    "jensenshannon": JensenShannon,
    "prdc": PRDC,
    "wasserstein": Wasserstein,
    "mmd": MMD,
    "authenticity": Authenticity,
    "domias": DOMIAS,
    "nndr": NNDR,
    "mia": MIA,
    "aia": AttributeInferenceAttack,
}


def evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_syn: pd.DataFrame,
    metrics: dict,
    discrete_features: list = [],
):
    """
    Quantitatively evaluates a bunch of metrics for a synthetic dataset w.r.t. a real test set.
    Returns a dictionary like {metric_name:result}.

    metrics: dict of dicts. Each metric has a dict containing parameters relevant to initializing that metric.
    """

    # one hot, label encode, standard scale
    X_tr_scaled, X_te_scaled, X_syn_scaled = preprocess_eval(
        X_train,
        X_test,
        X_syn,
        discrete_features=discrete_features,
        normalization="standard",
    )

    dict_ = {}
    for metric__ in metrics.keys():
        metric_ = metric__.split("-")[0].strip().lower()
        metric_cls = METRICS[metric_]
        # Use class properties to determine if discrete_features should be injected
        if getattr(metric_cls, "needs_discrete_features", False):
            metrics[metric__]["discrete_features"] = discrete_features
        metric = metric_cls(**metrics[metric__])
        # Use class property to determine which data to pass
        data_req = getattr(metric_cls, "data_requirement", None)
        if data_req == "test":
            metric_result = metric.evaluate(
                X_test,
                X_syn[-len(X_test) :],
            )
        elif data_req == "test_preprocessed":
            metric_result = metric.evaluate(
                X_te_scaled,
                X_syn_scaled[-len(X_test) :],
            )
        elif data_req == "train_preprocessed":
            metric_result = metric.evaluate(
                X_tr_scaled,
                X_syn_scaled[: len(X_train)],
            )
        elif data_req == "train_and_test":
            metric_result = metric.evaluate(X_train, X_test, X_syn)
        else:
            raise Exception(
                f"Metric {metric_} not (fully) implemented or missing data_requirement property"
            )

        # add result to dict (note that quantitative metrics have to output a dict, else they won't get added here)
        if type(metric_result) == dict:
            dict_.update(metric_result)

    return dict_
