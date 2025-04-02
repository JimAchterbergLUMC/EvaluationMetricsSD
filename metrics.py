import pandas as pd


def evaluate(
    X_train: pd.DataFrame, X_test: pd.DataFrame, X_syn: pd.DataFrame, metrics: list
):
    """
    Evaluates a bunch of metrics for a synthetic dataset w.r.t. a real test set.
    Returns a dictionary like {metric_name:result}.
    """
    dict_ = {}
    for metric_ in metrics:
        metric = _get_metric(metric_)
        if "domias" in metric_.lower():
            # only domias requires training data
            dict_[metric_] = metric(X_train, X_test, X_syn)
        else:
            dict_[metric_] = metric(X_test, X_syn)


def _get_metric(metric_name: str):
    pass


def jensen_shannon(real, syn):
    pass


def wasserstein(real, syn):
    pass


def precision_recall(real, syn):
    pass


def authenticity(real, syn):
    pass


def DOMIAS(train, test, syn):
    pass
