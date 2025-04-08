# computes metrics to construct a benchmark or evaluation report

import pandas as pd
import os
import shutil

from metrics import get_metric, marginal_plots, correlations, constraints
from utils import preprocess_eval


def evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_syn: pd.DataFrame,
    metrics: list,
    random_state: int,
):
    """
    Evaluates a bunch of metrics for a synthetic dataset w.r.t. a real test set.
    Returns a dictionary like {metric_name:result}.
    """
    # one hot, label encode, minmax scale
    X_tr_scaled, X_te_scaled, X_syn_scaled = preprocess_eval(
        X_train, X_test, X_syn, ohe_threshold=15
    )

    dict_ = {}
    for metric_ in metrics:
        metric = get_metric(metric_)
        if "domias" in metric_.lower():
            # only domias requires training data
            metric_result = metric(X_tr_scaled, X_te_scaled, X_syn_scaled, random_state)
        else:
            metric_result = metric(X_te_scaled, X_syn_scaled, random_state)
        if type(metric_result) == dict:
            dict_.update(metric_result)
        else:
            dict_[metric_] = metric_result

    return dict_


def report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    syn: pd.DataFrame,
    save_dir: str,
    random_state: int,
    **metric_params,
):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # compute evals and save as files in save_dir
    # -------------------------------------------------------
    # fidelity

    constraints(
        test,
        syn,
        f"{save_dir}/constraints",
        constraint_list=metric_params["constraints"],
    )

    marginal_plots(test, syn, f"{save_dir}/marginal_plots")

    correlations(test, syn, f"{save_dir}/correlations")

    # Association Rule Mining

    # Dimension Wise Prediction

    # PCA

    # tSNE

    # -------------------------------------------------------
    # privacy

    # MIA: make assumption regarding
