import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.stats import gaussian_kde
from umap.parametric_umap import ParametricUMAP
from sklearn.metrics import roc_auc_score


class DOMIAS:

    def __init__(
        self,
        ref_prop: float = 0.5,
        reduction: str = "umap",
        n_components: int = 5,
        random_state: int = 0,
    ):
        super().__init__()
        self.ref_prop = ref_prop
        self.reduction = reduction
        self.n_components = n_components
        self.random_state = random_state

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

        # dimensionality reduction through parametric UMAP
        if self.reduction == "umap":
            embedder = ParametricUMAP(
                n_components=self.n_components, random_state=self.random_state
            )
        # fit embedder on SD -> this typically has more samples and can thus learn a better embedding function
        synth_set = embedder.fit_transform(syn.to_numpy())
        reference_set = embedder.transform(reference_set.to_numpy())
        X_test = embedder.transform(X_test)

        kde = gaussian_kde(synth_set.T)
        P_G = kde(X_test.T)
        kde = gaussian_kde(reference_set.T)
        P_R = kde(X_test.T)
        P_rel = P_G / (P_R + 1e-10)
        P_rel = np.nan_to_num(P_rel)

        auc = roc_auc_score(Y_test, P_rel)

        return {"domias AUC": auc}


class Authenticity:

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
