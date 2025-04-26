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
    # ClusteringTest,
    CorrelationPlots,
    Projections,
    PRDC,
    Wasserstein,
    MMD,
)

from metrics.privacy import DOMIAS, Authenticity, NNDR

METRICS = {
    "domain_constraints": DomainConstraint,
    "arm": AssociationRuleMining,
    "dwp": DWP,
    "featurewise_plots": FeatureWisePlots,
    "classifier_test": ClassifierTest,
    # "clustering_test": ClusteringTest,
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

    # one hot, label encode, standard scale
    X_tr_scaled, X_te_scaled, X_syn_scaled = preprocess_eval(
        X_train, X_test, X_syn, ohe_threshold=15, normalization="standard"
    )

    dict_ = {}
    for metric__ in metrics.keys():
        metric_ = metric__.split("-")[0].strip()
        metric = METRICS[metric_](**metrics[metric__])
        if "domias" in metric_.lower():
            # domias requires training data and uses full SD set
            metric_result = metric.evaluate(X_tr_scaled, X_te_scaled, X_syn_scaled)
        elif "authenticity" in metric_.lower():
            # authenticity should be computed w.r.t. training data (so we also take that size synthetic data)
            metric_result = metric.evaluate(
                X_tr_scaled,
                X_syn_scaled[: len(X_train)],
            )
        elif "classifier_test" in metric_.lower():
            # classifier metric performs CV and thus preprocessing internally
            metric_result = metric.evaluate(
                X_train,
                X_syn[-len(X_test) :],
            )
        else:
            # other metrics are computed w.r.t. test set (so we also take that size synthetic data)
            metric_result = metric.evaluate(
                X_te_scaled,
                X_syn_scaled[-len(X_test) :],
            )

        if type(metric_result) == dict:
            dict_.update(metric_result)
        else:
            try:
                dict_[metric__] = metric_result
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
                test, syn[-len(test) :]
            )
            with open(f"{save_dir}/domain_constraints.txt", "w") as f:
                for c, vals in constraints.items():
                    f.write(f"{c} \n")
                    for d, val in vals.items():
                        f.write(f"{d}: {val} \n")

        elif key == "marginal_plots":
            metrics["marginal_plots"]["save_dir"] = f"{save_dir}/marginal_plots"
            plots = FeatureWisePlots(**metrics["marginal_plots"]).evaluate(
                test, syn[-len(test) :]
            )

        elif key == "correlation_plots":
            metrics["correlation_plots"]["save_dir"] = f"{save_dir}/correlation_plots"
            plots = CorrelationPlots(**metrics["correlation_plots"]).evaluate(
                test, syn[-len(test) :]
            )

        elif key == "arm":
            arm = AssociationRuleMining(**metrics["arm"]).evaluate(
                test, syn[-len(test) :]
            )
            with open(f"{save_dir}/association_rules.txt", "w") as f:
                for k, v in arm.items():
                    f.write(f"{k}: {v} \n")

        elif key == "dwp":
            dwp = DWP(**metrics["dwp"]).evaluate(test, syn[-len(test) :])
            with open(f"{save_dir}/dimensionwise_prediction.txt", "w") as f:
                for k, v in dwp.items():
                    f.write(f"{k}: \n")
                    for k_, v_ in v.items():
                        f.write(f"{k_}: {v_} \n")
        elif key == "projections":
            metrics["projections"]["save_dir"] = f"{save_dir}/projections"
            train_scaled, test_scaled, syn_scaled = preprocess_eval(
                train,
                test,
                syn[-len(test) :],
                ohe_threshold=15,
                normalization="standard",
            )
            projections = Projections(**metrics["projections"]).evaluate(
                test_scaled, syn_scaled
            )
        elif key == "nndr":
            # compute distances w.r.t. training set (in normalized space)
            metrics["nndr"]["save_dir"] = f"{save_dir}/nndr"
            train_scaled, test_scaled, syn_scaled = preprocess_eval(
                train,
                test,
                syn[: len(train)],
                ohe_threshold=15,
                normalization="standard",
            )
            nndr = NNDR(**metrics["nndr"]).evaluate(train_scaled, syn_scaled)

        else:
            raise Exception(f"metric {key} not implemented for reporting.")

    # -------------------------------------------------------
    # privacy
    # note that NNAA and authenticity should be w.r.t. training set

    # MIA: make assumption regarding
