import pandas as pd
from utils import preprocess_eval
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
import numpy as np
import torch
from geomloss import SamplesLoss
from sklearn import metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        metric = _get_metric(metric_)
        if "domias" in metric_.lower():
            # only domias requires training data
            dict_[metric_] = metric(X_train, X_test, X_syn, random_state)
        else:
            dict_[metric_] = metric(X_te_scaled, X_syn_scaled, random_state)

    return dict_


def _get_metric(metric_name: str):
    """
    Retrieves correct metric function, raises exception if provided metric is not implemented.
    """
    if metric_name.lower() == "mmd":
        return MMD
    elif metric_name.lower() == "wasserstein":
        return wasserstein
    elif metric_name.lower() == "precision-recall":
        return precision_recall
    elif metric_name.lower() == "authenticity":
        return authenticity
    elif metric_name.lower() == "domias":
        return DOMIAS
    else:
        raise Exception(
            "Metric not recognized. Make sure to provide metrics which are implemented."
        )


def MMD(real, syn, random_state):
    gamma = 1.0
    XX = metrics.pairwise.rbf_kernel(
        real.to_numpy().reshape(len(real), -1),
        real.to_numpy().reshape(len(real), -1),
        gamma,
    )
    YY = metrics.pairwise.rbf_kernel(
        syn.to_numpy().reshape(len(syn), -1),
        syn.to_numpy().reshape(len(syn), -1),
        gamma,
    )
    XY = metrics.pairwise.rbf_kernel(
        real.to_numpy().reshape(len(real), -1),
        syn.to_numpy().reshape(len(syn), -1),
        gamma,
    )
    score = XX.mean() + YY.mean() - 2 * XY.mean()
    return float(score)


def wasserstein(real: pd.DataFrame, syn: pd.DataFrame, random_state: int):
    # create torch tensors and send to device
    real_t = torch.from_numpy(real.to_numpy()).to(DEVICE)
    syn_t = torch.from_numpy(syn.to_numpy()).to(DEVICE)
    samplesloss = SamplesLoss(loss="sinkhorn")
    ws = (samplesloss(real_t, syn_t)).cpu().numpy().item()
    return ws


def precision_recall(real: pd.DataFrame, syn: pd.DataFrame, random_state: int):

    # compute precision recall

    pass


def authenticity(real: pd.DataFrame, syn: pd.DataFrame, random_state: int):

    # compute authenticity

    pass


def DOMIAS(
    train: pd.DataFrame, test: pd.DataFrame, syn: pd.DataFrame, random_state: int
):

    # factor decomposition

    # perform kde

    # compute domias

    pass
