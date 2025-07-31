import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.stats import gaussian_kde
from torch import nn
from umap import UMAP
from sklearn.metrics import (
    roc_auc_score,
    r2_score,
    root_mean_squared_error,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler, FunctionTransformer, LabelEncoder
from sklearn.decomposition import PCA
from xgboost import XGBClassifier, XGBRegressor
import os
from matplotlib import pyplot as plt
import seaborn as sns
from utils.utils import preprocess_prediction, preprocess_eval  # type: ignore
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins import Plugins

# TBD: implement more prediction models
CLF = {"xgb": XGBClassifier(max_depth=3)}
REG = {"xgb": XGBRegressor(max_depth=3)}


def precision_recall(
    y_true,
    y_pred,
):
    return precision_score(y_true, y_pred), recall_score(y_true, y_pred)


# TBD: implement more accuracy metrics
ACCURACY_METRICS = {
    "rmse": root_mean_squared_error,
    "r2": r2_score,
    "roc_auc": lambda y_true, y_score: roc_auc_score(
        y_true, y_score, average="micro", multi_class="ovr"
    ),
    "f1": f1_score,
    "acc": accuracy_score,
    "precision-recall": precision_recall,
}


class DOMIAS:
    data_requirement = "train_and_test"
    needs_discrete_features = True

    def __init__(
        self,
        ref_prop: float = 0.5,
        member_prop: float = 1.0,
        reduction: str = None,  # type: ignore
        n_neighbours: int = 5,
        n_components: int = 5,
        random_state: int = 0,
        metric: str = "roc_auc",
        discrete_features: list = [],
        quasi_identifiers: list = [],
        predict_top: float = 0.5,
    ):
        super().__init__()
        self.ref_prop = ref_prop
        self.member_prop = member_prop
        self.reduction = reduction
        self.n_components = n_components
        self.random_state = random_state
        self.n_neighbours = n_neighbours
        self.metric = metric
        self.discrete_features = discrete_features
        self.quasi_identifiers = quasi_identifiers
        self.predict_top = predict_top

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
        # grab only known QIs
        if len(self.quasi_identifiers) > 0:
            train, test, syn = (
                train[self.quasi_identifiers].copy(),  # type: ignore
                test[self.quasi_identifiers].copy(),  # type: ignore
                syn[self.quasi_identifiers].copy(),  # type: ignore
            )

        # preprocess before density estimation
        train, test, syn = preprocess_eval(
            train, test, syn, self.discrete_features, normalization="standard"
        )  # type: ignore

        members = train[: int(len(train) * self.member_prop)].copy()
        ref_size = int(self.ref_prop * len(test))
        reference_set, non_members = test[:ref_size], test[ref_size:]

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
        elif self.reduction is None:
            embedder = FunctionTransformer()
        else:
            raise Exception(f"Reduction {self.reduction} not implemented")
        # fit embedder on syn and reference to avoid leakage of test data which would inflate separability
        all_ = np.concatenate((syn.to_numpy(), reference_set.to_numpy()))
        all_ = embedder.fit_transform(all_)
        # project test data to same space
        all_ = np.concatenate((all_, embedder.transform(X_test)))  # type: ignore
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

        if self.metric != "roc_auc":
            threshold = np.percentile(P_rel, 100 * (1 - self.predict_top))
            # print(threshold)
            P_rel = P_rel > threshold

        score = ACCURACY_METRICS[self.metric](Y_test, P_rel)

        domias = {}
        if self.metric == "precision-recall":
            precision, recall = score
            domias[f"domias.metric=precision.reduction={self.reduction}"] = precision
            domias[f"domias.metric=recall.reduction={self.reduction}"] = recall
        else:
            domias[f"domias.metric={self.metric}.reduction={self.reduction}"] = score

        if self.metric != "roc_auc":
            # add a naive F1 score (e.g. predicting all as members)
            P_rel = np.ones_like(P_rel)
            naive_score = ACCURACY_METRICS[self.metric](Y_test, P_rel)
            if self.metric == "precision-recall":
                naive_precision, naive_recall = naive_score
                domias[f"domias.metric=precision.reduction={self.reduction}.naive"] = (
                    naive_precision
                )
                domias[f"domias.metric=recall.reduction={self.reduction}.naive"] = (
                    naive_recall
                )
            else:
                domias[
                    f"domias.metric={self.metric}.reduction={self.reduction}.naive"
                ] = naive_score

        return domias


class MIA:
    data_requirement = "train_and_test"
    needs_discrete_features = False

    def __init__(
        self,
        generator: str,
        generator_hparams: dict,
        random_state: int,
        metric: str = "roc_auc",
    ):
        self.generator = Plugins().get(generator, **generator_hparams)
        self.random_state = random_state
        self.metric = metric

    def evaluate(self, train: pd.DataFrame, test: pd.DataFrame, syn: pd.DataFrame):

        # fit a generator on the synthetic data
        syn = GenericDataLoader(syn, random_state=self.random_state)  # type: ignore
        self.generator.fit(syn)

        # generate "doubly" synthetic data
        syn = syn.dataframe()
        synsyn = self.generator.generate(len(syn))
        synsyn = synsyn.dataframe()
        # set labels: training data is the target class
        y = pd.Series(
            np.concatenate((np.zeros(len(synsyn)), np.ones(len(syn)))), name="target"
        )
        X = pd.concat([synsyn, syn])
        # train XGB classifier to distinguish training data from generated data
        for col in X.columns:
            try:
                X[col] = X[col].astype("float")
            except:
                X[col] = X[col].astype("category")
        clf = XGBClassifier(max_depth=3, tree_method="hist", enable_categorical=True)
        clf.fit(X, y)
        # make an attack dataset of members (training data) and non-members (test data)
        attack_set = pd.concat([train, test])
        for col in attack_set.columns:
            try:
                attack_set[col] = attack_set[col].astype("float")
            except:
                attack_set[col] = attack_set[col].astype("category")
        attack_y = pd.Series(
            np.concatenate((np.ones(len(train)), np.zeros(len(test)))), name="target"
        )
        # predict and score XGB classifier
        if self.metric == "roc_auc":
            attack_pred = clf.predict_proba(attack_set)
            attack_pred = attack_pred[:, 1]
        else:
            attack_pred = clf.predict(attack_set)

        score = ACCURACY_METRICS[self.metric](attack_y, attack_pred)
        results = {}
        if self.metric == "precision-recall":
            precision, recall = score
            results[f"mia.metric=precision"] = precision
            results[f"mia.metric=recall"] = recall
        else:
            results[f"mia.metric={self.metric}"] = score
        if self.metric == "f1":
            results[f"mia.metric={self.metric}.naive"] = ACCURACY_METRICS[self.metric](
                attack_y, np.ones(len(attack_y))
            )
        elif self.metric == "precision-recall":
            naive_precision, naive_recall = ACCURACY_METRICS[self.metric](
                attack_y, np.ones(len(attack_y))
            )
            results[f"mia.metric=precision.naive"] = naive_precision
            results[f"mia.metric=recall.naive"] = naive_recall
        return results


class Authenticity:
    data_requirement = "train_preprocessed"
    needs_discrete_features = False

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
    data_requirement = "test"
    needs_discrete_features = False

    # TBD: how to compare with "naive" score
    # TBD: use XGB native support for categorical features

    def __init__(
        self,
        quasi_identifiers: list,
        sensitive_attributes: list,
        model_name: str = "xgb",
        metric_numerical: str = "r2",
        metric_discrete: str = "roc_auc",
    ):
        self.quasi_identifiers = quasi_identifiers
        self.sensitive_attributes = sensitive_attributes
        self.model_name = model_name
        self.metric_numerical = metric_numerical
        self.metric_discrete = metric_discrete

    def evaluate(self, rd: pd.DataFrame, sd: pd.DataFrame):

        # select only qi's and sensitive features
        all_cols = self.quasi_identifiers + self.sensitive_attributes
        rd, sd = rd[all_cols], sd[all_cols]  # type: ignore

        if self.model_name == "xgb":
            # for xgb classifier use built-in functionality to handle categorical data
            x_rd = pd.DataFrame()
            x_sd = pd.DataFrame()
            self.discretes = []
            for col in rd.columns:
                try:
                    x_rd[col] = rd[col].astype("float")
                    x_sd[col] = sd[col].astype("float")
                except:
                    x_rd[col] = rd[col].astype("category")
                    x_sd[col] = sd[col].astype("category")
                    self.discretes.append(col)
        else:
            # TBD: add functionality for other classifiers
            raise Exception("No support yet for other models than XGB")

        scores = {}
        for col in x_rd.columns:
            # only take sensitive attributes as targets
            if col not in self.sensitive_attributes:
                continue

            # instantiate model and metrics
            if col in self.discretes:
                model = CLF[self.model_name]
                metric = ACCURACY_METRICS[self.metric_discrete]
            else:
                metric = ACCURACY_METRICS[self.metric_numerical]
                model = REG[self.model_name]

            # enable XGB native categorical support
            if self.model_name == "xgb":
                model.set_params(tree_method="hist", enable_categorical=True)

            X_tr, X_te = (
                x_rd[self.quasi_identifiers].copy(),
                x_sd[self.quasi_identifiers].copy(),
            )
            y_tr, y_te = (
                x_rd[col].copy(),
                x_sd[col].copy(),
            )

            # numerically encode categorical labels
            if col in self.discretes:
                encoder = LabelEncoder()
                y_tr = encoder.fit_transform(y_tr)
                y_te = encoder.transform(y_te)

            model.fit(X_tr, y_tr)
            preds = self._predict(y_tr, X_te, col, model)
            score = metric(y_te, preds)
            scores[
                f"aia.target={col}.metric={'roc_auc' if col in self.discretes else self.metric_numerical}"
            ] = score
        return scores

    def _predict(self, y_tr, X_test, target, model):
        if target in self.discretes and self.metric_discrete == "roc_auc":
            if len(np.unique(y_tr)) == 2:
                return model.predict_proba(X_test)[:, 1]
            else:
                return model.predict_proba(X_test)
        else:
            return model.predict(X_test)


class NNDR:
    data_requirement = "train_and_test_preprocessed"
    needs_discrete_features = False

    def __init__(
        self,
        metric: str = "euclidean",
        percentiles: list = [1, 2, 5, 10],
    ):
        self.metric = metric
        self.percentiles = percentiles

    def evaluate(self, rd_train: pd.DataFrame, rd_test: pd.DataFrame, sd: pd.DataFrame):
        nndr_train = self._compute_nndr(rd_train, sd)
        nndr_test = self._compute_nndr(rd_test, sd)

        results = {}
        for p in self.percentiles:
            val_train = np.percentile(nndr_train, p)
            val_test = np.percentile(nndr_test, p)
            results[f"nndr.percentile={p}.train"] = val_train
            results[f"nndr.percentile={p}.test"] = val_test
            results[f"nndr.percentile={p}.ratio"] = val_train / val_test
        return results

    def _compute_nndr(self, rd: pd.DataFrame, sd: pd.DataFrame):
        # calculate distance from sd to nearest rd -> normalized by distance to next nearest rd
        nn = NearestNeighbors(n_neighbors=2, metric=self.metric)
        nn.fit(rd)
        distances, indices = nn.kneighbors(sd)
        nearest_dist = distances[:, 0]
        second_nearest_dist = distances[:, 1]
        nndr = nearest_dist / second_nearest_dist
        return nndr
