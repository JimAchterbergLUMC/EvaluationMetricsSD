import pandas as pd
from utils import preprocess_eval
import numpy as np
import torch
from geomloss import SamplesLoss
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors

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
            # metrics are computed with preprocessed data (scaled, one hot, and label encoded)
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


def MMD(real: pd.DataFrame, syn: pd.DataFrame, random_state: int, gamma=1.0):
    """
    Computes MMD with RBF kernel.
    Code adapted from Synthcity library.
    """

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
    """
    Calculates Wasserstein distance (Sinkhorn divergence approximation).
    Code adapted from Synthcity.
    """
    # create torch tensors and send to device
    real_t = torch.from_numpy(real.to_numpy()).to(DEVICE)
    syn_t = torch.from_numpy(syn.to_numpy()).to(DEVICE)
    samplesloss = SamplesLoss(loss="sinkhorn")
    ws = (samplesloss(real_t, syn_t)).cpu().numpy().item()
    return ws


def precision_recall(
    real: pd.DataFrame, syn: pd.DataFrame, random_state: int, nearest_k: int = 5
):
    """
    Computes precision-recall for distributions using Naeem et al.'s density estimation.
    Code adapted from Synthcity.
    """

    def _compute_pairwise_distance(
        data_x: np.ndarray, data_y: np.ndarray = None
    ) -> np.ndarray:
        if data_y is None:
            data_y = data_x

        dists = metrics.pairwise_distances(data_x, data_y)
        return dists

    def _get_kth_value(unsorted: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
        indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
        k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
        kth_values = k_smallests.max(axis=axis)
        return kth_values

    def _compute_nearest_neighbour_distances(
        input_features: np.ndarray, nearest_k: int
    ) -> np.ndarray:
        distances = _compute_pairwise_distance(input_features)
        radii = _get_kth_value(distances, k=nearest_k + 1, axis=-1)
        return radii

    real_nearest_neighbour_distances = _compute_nearest_neighbour_distances(
        real.to_numpy(), nearest_k
    )
    fake_nearest_neighbour_distances = _compute_nearest_neighbour_distances(
        syn.to_numpy(), nearest_k
    )
    distance_real_fake = _compute_pairwise_distance(real.to_numpy(), syn.to_numpy())

    precision = (
        (distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1))
        .any(axis=0)
        .mean()
    )

    recall = (
        (distance_real_fake < np.expand_dims(fake_nearest_neighbour_distances, axis=0))
        .any(axis=1)
        .mean()
    )
    return (precision, recall)


def authenticity(real: pd.DataFrame, syn: pd.DataFrame, random_state: int):
    """
    Computes authenticity score as described in Alaa et al.
    Code adapted from Synthcity.
    """

    nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(real)
    real_to_real, _ = nbrs_real.kneighbors(real)

    nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(syn)
    real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(real)

    real_to_real = real_to_real[:, 1].squeeze()
    real_to_synth = real_to_synth.squeeze()
    real_to_synth_args = real_to_synth_args.squeeze()

    authen = real_to_real[real_to_synth_args] < real_to_synth

    return np.mean(authen)


def DOMIAS(
    train: pd.DataFrame, test: pd.DataFrame, syn: pd.DataFrame, random_state: int
):
    """
    Computes DOMIAS membership inference attack accuracy.
    Estimates density through Gaussian KDE after factor decomposition.
    Code based on Synthcity library.
    """

    # factor decomposition

    # perform kde

    # compute domias

    pass
