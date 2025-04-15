import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.stats import gaussian_kde
from torch import nn
from umap import UMAP
from sklearn.metrics import roc_auc_score, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier, XGBRegressor

from utils import preprocess_prediction

# TBD: implement more prediction models
CLF = {"xgb": XGBClassifier(max_depth=3)}
REG = {"xgb": XGBRegressor(max_depth=3)}

# TBD: implement more accuracy metrics
ACCURACY_METRICS = {
    "rmse": root_mean_squared_error,
    "roc_auc": lambda y_true, y_score: roc_auc_score(
        y_true, y_score, average="micro", multi_class="ovr"
    ),
}


class DOMIAS:

    def __init__(
        self,
        ref_prop: float = 0.5,
        reduction: str = "umap",
        n_neighbours: int = 5,
        n_components: int = 5,
        random_state: int = 0,
    ):
        super().__init__()
        self.ref_prop = ref_prop
        self.reduction = reduction
        self.n_components = n_components
        self.random_state = random_state
        self.n_neighbours = n_neighbours

    def evaluate(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        syn: pd.DataFrame,
    ):
        """
        Computes DOMIAS membership inference attack accuracy (AUROC).
        Estimates density through Gaussian KDE after dimensionality reduction.
        Code based on Synthcity library.

        ref_prop: proportion of test set used as reference set for computing RD density.
        n_components: number of dimensions to retain after reduction.
        """
        members = train.copy()
        ref_size = int(self.ref_prop * len(test))
        non_members, reference_set = test[:ref_size], test[-ref_size:]

        X_test = np.concatenate((members.to_numpy(), non_members.to_numpy()))
        Y_test = np.concatenate(
            [np.ones(members.shape[0]), np.zeros(non_members.shape[0])]
        ).astype(bool)

        # dimensionality reduction
        if self.reduction == "umap":
            embedder = UMAP(
                n_neighbors=self.n_neighbours,
                n_components=self.n_components,
                random_state=self.random_state,
            )
        elif self.reduction == "pca":
            embedder = PCA(
                n_components=self.n_components, random_state=self.random_state
            )
        else:
            raise Exception(f"Reduction {self.reduction} not implemented")
        # fit embedder on syn and reference to avoid leakage of test data which would inflate separability
        all_ = np.concatenate((syn.to_numpy(), reference_set.to_numpy()))
        all_ = embedder.fit_transform(all_)
        # project test data to same space
        all_ = np.concatenate((all_, embedder.transform(X_test)))
        # standardize for gaussian KDE
        all_ = StandardScaler().fit_transform(all_)
        synth_set = all_[: len(syn)]
        reference_set = all_[len(syn) : len(syn) + len(reference_set)]
        X_test = all_[-len(X_test) :]

        kde = gaussian_kde(synth_set.T)
        P_G = kde(X_test.T)
        kde = gaussian_kde(reference_set.T)
        P_R = kde(X_test.T)
        P_rel = P_G / (P_R + 1e-10)
        P_rel = np.nan_to_num(P_rel)

        auc = roc_auc_score(Y_test, P_rel)

        return {f"domias AUC ({self.reduction})": auc}


class Authenticity:
    """
    Note that authenticity should be computed w.r.t. training set.
    """

    def __init__(self):
        super().__init__()

    def evaluate(
        self,
        rd: pd.DataFrame,
        sd: pd.DataFrame,
    ):
        nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(rd)
        real_to_real, _ = nbrs_real.kneighbors(rd)

        nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(sd)
        real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(rd)

        real_to_real = real_to_real[:, 1].squeeze()
        real_to_synth = real_to_synth.squeeze()
        real_to_synth_args = real_to_synth_args.squeeze()

        authen = real_to_real[real_to_synth_args] < real_to_synth

        return {"authenticity": np.mean(authen)}


class AttributeInferenceAttack:

    # TBD: how to compare with "naive" score

    def __init__(
        self,
        quasi_identifiers: list,
        sensitive_attributes: list,
        discrete_features: list,
        model_name: str = "xgb",
        metric_numerical: str = "rmse",
        metric_discrete: str = "roc_auc",
    ):
        self.quasi_identifiers = quasi_identifiers
        self.sensitive_attributes = sensitive_attributes
        self.discrete_features = discrete_features
        self.model_name = model_name
        self.metric_numerical = metric_numerical
        self.metric_discrete = metric_discrete

    def evaluate(self, rd: pd.DataFrame, sd: pd.DataFrame):

        # returns dict: {sensitive feature: accuracy}
        all_cols = self.quasi_identifiers + self.sensitive_attributes
        dict_ = {}
        for target in self.sensitive_attributes:
            all_cols = self.quasi_identifiers + [target]
            X_tr, y_tr, X_te, y_te = preprocess_prediction(
                train=sd[all_cols],
                test=rd[all_cols],
                target_col=target,
                discrete_features=self.discrete_features,
                normalization="standard",
            )
            if target in self.discrete_features:
                model = CLF[self.model_name]
                metric = ACCURACY_METRICS[self.metric_discrete]
            else:
                model = REG[self.model_name]
                metric = ACCURACY_METRICS[self.metric_numerical]

            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            if target in self.discrete_features:
                if self.metric_discrete == "roc_auc":
                    preds = model.predict_proba(X_te)
                    if len(np.unique(y_tr)) == 2:
                        preds = preds[:, 1]
                else:
                    preds = model.predict(X_te)
            else:
                preds = model.predict(X_te)
            dict_[target] = metric(y_te, preds)
        return dict_


class NNAA:
    def __init__(self, metric="euclidean"):
        pass

    def evaluate(self, rd: pd.DataFrame, sd: pd.DataFrame):
        pass
