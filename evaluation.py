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
        # add discrete features to metric_params for those which need it
        if metric_ in [
            "featurewise_plots",
            "correlation_matrices",
            "association_rules",
            "domias",
        ]:
            metrics[metric__]["discrete_features"] = discrete_features
        metric = METRICS[metric_](**metrics[metric__])

        # metrics computed w.r.t. test set in original feature space
        if metric_ in [
            "domain_constraints",
            "association_rules",
            "dwp",
            "classifier_test",
            "featurewise_plots",
            "correlation_matrices",
            "aia",
        ]:
            metric_result = metric.evaluate(
                X_test,
                X_syn[-len(X_test) :],
            )
        # metrics computed w.r.t. preprocessed test set (mostly distance-based fidelity measures)
        elif metric_ in ["prdc", "wasserstein", "mmd", "projections", "jensenshannon"]:
            metric_result = metric.evaluate(
                X_te_scaled,
                X_syn_scaled[-len(X_test) :],
            )
        # metrics computed w.r.t. preprocessed train set (mostly distance-based privacy measures)
        elif metric_ in ["authenticity", "nndr"]:
            metric_result = metric.evaluate(
                X_tr_scaled,
                X_syn_scaled[: len(X_train)],
            )
        # metrics computed w.r.t. train ánd test set (mostly MIAs)
        elif metric_ in ["domias", "mia"]:
            metric_result = metric.evaluate(X_train, X_test, X_syn)
        else:
            raise Exception(f"Metric {metric_} not (fully) implemented")

        # add result to dict (note that quantitative metrics have to output a dict, else they won't get added here)
        if type(metric_result) == dict:
            dict_.update(metric_result)

    return dict_
