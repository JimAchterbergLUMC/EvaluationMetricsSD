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


def report(
    train: pd.DataFrame, test: pd.DataFrame, syn: pd.DataFrame, save_dir: str, **metrics
):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    for key in metrics.keys():

        if key == "domain_constraints":
            constraints = DomainConstraint(**metrics["domain_constraints"]).evaluate(
                test, syn
            )  # output: dict(constraint1:{Real:...,Synthetic:...},constraint2:{Real:...,Synthetic:...})
            # write dict to text file
            with open(f"{save_dir}/domain_constraints.txt", "w") as f:
                for c, vals in constraints.items():
                    f.write(f"{c} \n")
                    for d, val in vals.items():
                        f.write(f"{d}: {val} \n")

        elif key == "marginal_plots":
            metrics["marginal_plots"]["save_dir"] = f"{save_dir}/marginal_plots"
            plots = FeatureWisePlots(**metrics["marginal_plots"]).evaluate(test, syn)

        elif key == "correlation_plots":
            metrics["correlation_plots"]["save_dir"] = f"{save_dir}/correlation_plots"
            plots = CorrelationPlots(**metrics["correlation_plots"]).evaluate(test, syn)

        elif key == "arm":
            arm = AssociationRuleMining(**metrics["arm"]).evaluate(test, syn)
            with open(f"{save_dir}/association_rules.txt", "w") as f:
                for k, v in arm.items():
                    f.write(f"{k}: {v} \n")

        elif key == "dwp":
            dwp = DWP(**metrics["dwp"]).evaluate(test, syn)
            with open(f"{save_dir}/dimensionwise_prediction.txt", "w") as f:
                for k, v in dwp.items():
                    f.write(f"{k}: \n")
                    for k_, v_ in v.items():
                        f.write(f"{k_}: {v_} \n")

    # projections = Projections(**metrics["projections"])

    # -------------------------------------------------------
    # privacy

    # MIA: make assumption regarding
