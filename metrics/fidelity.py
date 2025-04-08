from typing import Any
import matplotlib.axes
import numpy as np
import matplotlib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
from torchvision import transforms as transforms
from scipy.spatial.distance import jensenshannon
from geomloss import SamplesLoss
import torch
from sklearn import metrics
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    KBinsDiscretizer,
    StandardScaler,
)
from mlxtend.frequent_patterns import apriori, association_rules
import seaborn as sns


class DomainConstraint:

    def __init__(self, constraint: str):
        super().__init__()
        self.constraint = constraint

    def evaluate(self, rd: pd.DataFrame, sd: pd.DataFrame):
        result = {}
        result["Real"] = rd.eval(self.constraint).mean()
        result["Synthetic"] = sd.eval(self.constraint).mean()
        return result


class FeatureWisePlots:

    def __init__(
        self,
        discrete_features: list,
        plot_cols: int = 3,
        figsize: tuple = (10, 10),
    ):
        super().__init__()
        self.discrete_features = discrete_features
        self.plot_cols = plot_cols
        self.figsize = figsize

    def evaluate(self, rd: pd.DataFrame, sd: pd.DataFrame):
        self.numerical_features = [
            x for x in rd.columns if x not in self.discrete_features
        ]

        n_features = len(self.numerical_features) + len(self.discrete_features)
        n_rows = -(-n_features // self.plot_cols)  # Compute rows with ceiling division

        fig, axes = plt.subplots(n_rows, self.plot_cols, figsize=self.figsize)
        axes = axes.flatten()  # Flatten in case of a single row

        for i, feature in enumerate(self.numerical_features + self.discrete_features):
            ax = axes[i]

            if feature in self.numerical_features:
                sns.histplot(
                    rd[feature],
                    kde=True,
                    stat="density",
                    bins=30,
                    label="Real",
                    color="blue",
                    alpha=0.3,
                    ax=ax,
                )
                sns.histplot(
                    sd[feature],
                    kde=True,
                    stat="density",
                    bins=30,
                    label="Synthetic",
                    color="red",
                    alpha=0.3,
                    ax=ax,
                )

                # Compute statistics
                rd_mean, rd_std = rd[feature].mean(), rd[feature].std()
                sd_mean, sd_std = sd[feature].mean(), sd[feature].std()

                # Annotate with mean and std
                ax.text(
                    0.05,
                    0.95,
                    f"μ={rd_mean:.2f}, σ={rd_std:.2f}\n",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="blue",
                    verticalalignment="top",
                    # bbox=dict(facecolor="white", alpha=0.5),
                )
                ax.text(
                    0.05,
                    0.85,
                    f"μ={sd_mean:.2f}, σ={sd_std:.2f}\n",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="red",
                    verticalalignment="top",
                    # bbox=dict(facecolor="white", alpha=0.5),
                )
            else:
                rd_counts = (
                    rd[feature].value_counts(normalize=True).sort_index() * 100
                )  # Convert to percentage
                sd_counts = sd[feature].value_counts(normalize=True).sort_index() * 100

                categories = sorted(set(rd_counts.index).union(sd_counts.index))
                rd_counts = rd_counts.reindex(categories, fill_value=0)
                sd_counts = sd_counts.reindex(categories, fill_value=0)

                plot_data = pd.DataFrame(
                    {"Category": categories, "Real": rd_counts, "Synthetic": sd_counts}
                ).melt(id_vars=["Category"])
                sns.barplot(
                    x="Category",
                    y="value",
                    hue="variable",
                    data=plot_data,
                    ax=ax,
                    palette=["blue", "red"],
                    alpha=0.5,
                )
                ax.set_ylabel("Proportion (%)")

                # Annotate with category percentages
                annot_real = "\n".join(
                    [f"{cat}: {rd_counts[cat]:.1f}%" for cat in categories]
                )
                annot_syn = "\n".join([f"{sd_counts[cat]:.1f}%" for cat in categories])

                ax.text(
                    0.05,
                    0.95,
                    annot_real,
                    color="blue",
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    # bbox=dict(facecolor="white", alpha=0.5),
                )

                ax.text(
                    0.55,
                    0.95,
                    annot_syn,
                    color="red",
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    # bbox=dict(facecolor="white", alpha=0.5),
                )

            ax.set_xlabel("")
            ax.set_title(feature)

        # Remove unused subplots if any
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        legend = axes[-2].legend()
        axes[-2].get_legend().remove()

        fig.legend(
            legend.legend_handles,
            [t.get_text() for t in legend.get_texts()],
            loc="lower right",
            fontsize=12,
        )

        plt.tight_layout()
        return plt


class CorrelationPlots:
    def __init__(
        self,
        discrete_features: list,
        figsize: tuple = (15, 10),
    ):
        super().__init__()
        self.discrete_features = discrete_features
        self.figsize = figsize

    def evaluate(self, rd: pd.DataFrame, sd: pd.DataFrame):
        self.numerical_features = [
            x for x in rd.columns if x not in self.discrete_features
        ]

        corr_rd = self.compute_mixed_correlation_matrix(rd)
        corr_sd = self.compute_mixed_correlation_matrix(sd)

        fig, axs = plt.subplots(ncols=2, figsize=self.figsize)
        sns.heatmap(
            corr_rd,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            linewidths=0.5,
            ax=axs[0],
            square=True,
            cbar_kws={"shrink": 0.5},
            vmin=-1,
            vmax=1,
        )
        axs[0].set_title("Real")
        sns.heatmap(
            corr_sd,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            linewidths=0.5,
            square=True,
            ax=axs[1],
            cbar_kws={"shrink": 0.5},
            vmin=-1,
            vmax=1,
        )
        axs[1].set_title("Synthetic")
        plt.tight_layout()
        return plt

    def compute_mixed_correlation_matrix(self, data):

        corr_matrix = pd.DataFrame(
            index=data.columns, columns=data.columns, dtype=float
        )

        for i, f1 in enumerate(data.columns):
            for j, f2 in enumerate(data.columns):
                if j < i:
                    continue  # Only compute upper triangular part

                if f1 in self.numerical_features and f2 in self.numerical_features:
                    corr_matrix.loc[f1, f2] = stats.spearmanr(data[f1], data[f2])[0]
                elif f1 in self.discrete_features and f2 in self.discrete_features:
                    corr_matrix.loc[f1, f2] = self.cramers_v(data[f1], data[f2])
                elif f1 in self.numerical_features and f2 in self.discrete_features:
                    corr_matrix.loc[f1, f2] = self.correlation_ratio(data[f2], data[f1])
                elif f1 in self.discrete_features and f2 in self.numerical_features:

                    corr_matrix.loc[f1, f2] = self.correlation_ratio(data[f1], data[f2])

                corr_matrix.loc[f2, f1] = corr_matrix.loc[
                    f1, f2
                ]  # Fill symmetric value

        return corr_matrix.astype(float)

    # create correlation heatmap
    def cramers_v(self, x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        return np.sqrt(phi2 / min(k - 1, r - 1))

    def correlation_ratio(self, categories, values):
        categories = np.array(categories)
        values = np.array(values)

        class_means = [
            values[categories == cat].mean() for cat in np.unique(categories)
        ]
        overall_mean = values.mean()

        numerator = np.sum(
            [
                len(values[categories == cat]) * (class_mean - overall_mean) ** 2
                for cat, class_mean in zip(np.unique(categories), class_means)
            ]
        )
        denominator = np.sum((values - overall_mean) ** 2)

        return numerator / denominator if denominator != 0 else 0


class AssociationRuleMining:

    def __init__(
        self,
        discrete_features: list,
        n_bins: int = 4,
        min_support: float = 0.2,
        min_confidence: float = 0.7,
    ):
        super().__init__()
        self.discrete_features = discrete_features
        self.n_bins = n_bins
        self.min_support = min_support
        self.min_confidence = min_confidence

    def evaluate(self, rd: pd.DataFrame, sd: pd.DataFrame):
        # discretize datasets (one hot encoded items)
        rd, sd = self._discretize(rd), self._discretize(sd)

        # find association rules
        rd_rules, sd_rules = self._rule_mining(rd), self._rule_mining(sd)

        # get precision/recall
        precision, recall = self._precision_recall(sd_rules, rd_rules)

        return {
            "#Real rules": len(rd_rules),
            "#Synthetic rules": len(sd_rules),
            "Precision": precision,
            "Recall": recall,
        }

    def _rule_mining(
        self,
        df: pd.DataFrame,
    ):
        frequent_itemsets = apriori(df, min_support=self.min_support, use_colnames=True)
        rules = association_rules(
            frequent_itemsets, metric="confidence", min_threshold=self.min_confidence
        )
        return rules

    def _discretize(self, data: pd.DataFrame):
        df_discretized = []
        for col in data.columns:
            if col not in self.discrete_features:
                if self.n_bins == 2:
                    encode = "ordinal"
                else:
                    encode = "onehot-dense"
                discretizer = KBinsDiscretizer(
                    n_bins=self.n_bins, encode=encode, strategy="uniform"
                )
                df_discretized.append(
                    pd.DataFrame(
                        discretizer.fit_transform(data[[col]]),
                        columns=discretizer.get_feature_names_out(),
                    )
                )
            else:
                if data[col].nunique() == 2:
                    discretizer = LabelEncoder()
                    df_discretized.append(
                        pd.DataFrame(
                            discretizer.fit_transform(data[[col]]),
                            columns=[col],
                        )
                    )
                else:
                    discretizer = OneHotEncoder(sparse_output=False)
                    df_discretized.append(
                        pd.DataFrame(
                            discretizer.fit_transform(data[[col]]),
                            columns=discretizer.get_feature_names_out(),
                        )
                    )

        df_discretized = pd.concat(df_discretized, axis=1)
        return df_discretized

    def _precision_recall(self, rules_set1, rules_set2):
        set1 = rules_set1[["antecedents", "consequents"]].drop_duplicates()
        set2 = rules_set2[["antecedents", "consequents"]].drop_duplicates()

        # Find common rules (intersection of the two sets)
        common_rules = pd.merge(
            set1, set2, on=["antecedents", "consequents"], how="inner"
        )

        # Precision: how many of the rules in Set1 are also in Set2
        precision = len(common_rules) / len(set1) if len(set1) > 0 else 0

        # Recall: how many of the rules in Set2 are also in Set1
        recall = len(common_rules) / len(set2) if len(set2) > 0 else 0

        return precision, recall


class DWP:

    def __init__(
        self,
        discrete_features,
        reg=RandomForestRegressor(),
        clf=RandomForestClassifier(),
        metric_numerical=root_mean_squared_error,
        metric_discrete=lambda y_true, y_score: roc_auc_score(
            y_true, y_score, average="micro", multi_class="ovr"
        ),
        k: int = 5,
    ):
        super().__init__()
        self.discrete_features = discrete_features
        self.reg = reg
        self.clf = clf
        self.metric_numerical = metric_numerical
        self.metric_discrete = metric_discrete
        self.k = k

    def onehot(self, data: pd.DataFrame):
        # separate original discrete features as to not change it when looping through target features
        self.discrete_features_new = self.discrete_features.copy()
        df = []
        for col in data.columns:
            if col in self.discrete_features:
                encoder = OneHotEncoder(sparse_output=False, drop="if_binary")
                new_data = encoder.fit_transform(data[[col]])
                new_cols = (
                    encoder.get_feature_names_out(input_features=[col])
                    if len(np.unique(data[[col]])) > 2
                    else [col]
                )
                # remove old col name and add new ohe col names
                self.discrete_features_new.remove(col)
                self.discrete_features_new.extend(new_cols)
                df.append(pd.DataFrame(new_data, columns=new_cols))
            else:
                df.append(data[[col]])
        df = pd.concat(df, axis=1)
        return df

    def evaluate(self, rd: pd.DataFrame, sd: pd.DataFrame):

        results_rd = []
        results_sd = []
        cv = KFold(n_splits=self.k)
        for idx_tr, idx_te in cv.split(rd):
            rd_train, rd_test = rd.iloc[idx_tr], rd.iloc[idx_te]
            sd_train = sd.iloc[idx_tr]

            results_rd.append(self.dwp(train=rd_train, test=rd_test))
            results_sd.append(self.dwp(train=sd_train, test=rd_test))
        results = {}
        results["Real"] = pd.DataFrame(results_rd).mean().to_dict()
        results["Synthetic"] = pd.DataFrame(results_sd).mean().to_dict()
        return results

    def dwp(self, train: pd.DataFrame, test: pd.DataFrame):
        dwp = {}
        for col in train.columns:
            if col in self.discrete_features:
                model = self.clf
                metric = self.metric_discrete
            else:
                model = self.reg
                metric = self.metric_numerical
            X_tr = train.drop(col, axis=1)
            y_tr = train[col]
            X_te = test.drop(col, axis=1)
            y_te = test[col]

            # perform preprocessing (OHE and scaling)
            xx = self.onehot(data=pd.concat([X_tr, X_te], ignore_index=True))
            # get numerical feature names (after changes due to OHE)
            numerical_features = [
                x for x in X_tr.columns if x not in self.discrete_features_new
            ]
            X_tr, X_te = xx.iloc[: len(X_tr)], xx.iloc[len(X_tr) :]
            scaler = StandardScaler()
            X_tr[numerical_features] = scaler.fit_transform(X_tr[numerical_features])
            X_te[numerical_features] = scaler.transform(X_te[numerical_features])

            if col not in self.discrete_features:
                # normalize numerical targets
                scaler = StandardScaler()
                y_tr = pd.Series(
                    scaler.fit_transform(y_tr.to_frame()).flatten(), name=y_tr.name
                )
                y_te = pd.Series(
                    scaler.transform(y_te.to_frame()).flatten(), name=y_te.name
                )
            else:
                # label encode discrete targets
                scaler = LabelEncoder()
                y_tr = pd.Series(scaler.fit_transform(y_tr), name=y_tr.name)
                y_te = pd.Series(scaler.transform(y_te), name=y_te.name)

            model.fit(X_tr, y_tr)

            if col in self.discrete_features:
                preds = model.predict_proba(X_te)
            else:
                preds = model.predict(X_te)
            # for binary rocauc
            if (col in self.discrete_features) and (len(np.unique(y_tr)) == 2):
                preds = preds[:, 1]
            score = metric(y_te, preds)
            dwp.update({col: score})
        return dwp


class Projections:

    def __init__(
        self,
        ax: matplotlib.axes.Axes,
        embedder: Any = PCA(n_components=2),
        title: str = "Projection",
        **plot_kwargs,
    ):
        super().__init__()
        self.ax = ax
        self.embedder = embedder
        self.title = title
        self.plot_kwargs = plot_kwargs

    def evaluate(
        self,
        rd: pd.DataFrame,
        sd: pd.DataFrame,
    ):
        """
        Expects preprocessed data.
        """
        rd, sd = rd.to_numpy(), sd.to_numpy()
        data = np.concatenate((rd, sd))
        emb = self.embedder.fit_transform(data)
        rd_emb = emb[: len(rd)]
        sd_emb = emb[len(rd) :]

        sns.scatterplot(
            x=rd_emb[:, 0],
            y=rd_emb[:, 1],
            color="blue",
            ax=self.ax,
            label="Real",
            **self.plot_kwargs,
        )
        sns.scatterplot(
            x=sd_emb[:, 0],
            y=sd_emb[:, 1],
            color="red",
            ax=self.ax,
            label="Synthetic",
            **self.plot_kwargs,
        )
        self.ax.set_title(self.title)


class ClassifierTest:

    def __init__(
        self,
        discrete_features: list,
        clf: Any = RandomForestClassifier(),
        kfolds: int = 5,
    ):
        super().__init__()
        self.clf = clf
        self.kfolds = kfolds
        self.discrete_features = discrete_features

    def onehot(self, data: pd.DataFrame):

        df = []
        for col in data.columns:
            if col in self.discrete_features:
                encoder = OneHotEncoder(sparse_output=False, drop="if_binary")
                new_data = encoder.fit_transform(data[[col]])
                new_cols = (
                    encoder.get_feature_names_out(input_features=[col])
                    if len(np.unique(data[[col]])) > 2
                    else [col]
                )
                # remove old col name and add new ohe col names
                self.discrete_features.remove(col)
                self.discrete_features.extend(new_cols)
                df.append(pd.DataFrame(new_data, columns=new_cols))
            else:
                df.append(data[[col]])
        df = pd.concat(df, axis=1)
        return df

    def evaluate(
        self,
        rd: pd.DataFrame,
        sd: pd.DataFrame,
    ):
        """
        Expects non-preprocessed data.
        """
        # onehot encode rd and sd together to avoid issues arising from non-occuring categories
        X = self.onehot(pd.concat([rd, sd], ignore_index=True))
        numerical_features = [x for x in X.columns if x not in self.discrete_features]
        numerical_features_idx = [X.columns.get_loc(col) for col in numerical_features]
        X = X.to_numpy()
        y = np.concatenate((np.zeros(len(rd)), np.ones(len(sd))))

        skf = StratifiedKFold(n_splits=self.kfolds, shuffle=True, random_state=0)
        scores = []
        for idx_tr, idx_te in skf.split(X, y):
            X_tr, X_te = X[idx_tr], X[idx_te]
            # fit scaler only on train data to avoid information leakage
            scaler = StandardScaler()
            X_tr[:, numerical_features_idx] = scaler.fit_transform(
                X_tr[:, numerical_features_idx]
            )
            X_te[:, numerical_features_idx] = scaler.transform(
                X_te[:, numerical_features_idx]
            )
            y_tr, y_te = y[idx_tr], y[idx_te]

            self.clf.fit(X_tr, y_tr)
            preds = self.clf.predict_proba(X_te)
            scores.append(roc_auc_score(y_te, preds[:, 1]))

        return {f"AUC (avg. over {self.kfolds} folds)": np.mean(scores)}


class ClusteringTest:

    def __init__(
        self,
        discrete_features: list,
        cls: Any = KMeans(n_clusters=5),
    ):
        super().__init__()
        self.discrete_features = discrete_features
        self.cls = cls

    def onehot(self, data: pd.DataFrame):

        df = []
        for col in data.columns:
            if col in self.discrete_features:
                encoder = OneHotEncoder(sparse_output=False, drop="if_binary")
                new_data = encoder.fit_transform(data[[col]])
                new_cols = (
                    encoder.get_feature_names_out(input_features=[col])
                    if len(np.unique(data[[col]])) > 2
                    else [col]
                )
                # remove old col name and add new ohe col names
                self.discrete_features.remove(col)
                self.discrete_features.extend(new_cols)
                df.append(pd.DataFrame(new_data, columns=new_cols))
            else:
                df.append(data[[col]])
        df = pd.concat(df, axis=1)
        return df

    def evaluate(
        self,
        rd: pd.DataFrame,
        sd: pd.DataFrame,
    ):

        X = self.onehot(pd.concat([rd, sd], ignore_index=True))
        numerical_features = [x for x in X.columns if x not in self.discrete_features]
        X[numerical_features] = StandardScaler().fit_transform(X[numerical_features])
        X = X.to_numpy()

        clusters = self.cls.fit_predict(X)
        clusters_rd = clusters[: len(rd)]
        clusters_sd = clusters[len(rd) :]
        rd_counts = np.bincount(clusters_rd, minlength=len(np.unique(clusters)))
        sd_counts = np.bincount(clusters_sd, minlength=len(np.unique(clusters)))

        ratios = {
            cluster: (
                sd_counts[cluster] / rd_counts[cluster]
                if rd_counts[cluster] > 0
                else np.inf
            )
            for cluster in np.unique(clusters)
        }
        score = np.mean(
            (np.fromiter(ratios.values(), dtype=float) - (len(sd) / len(rd))) ** 2
        )
        return {"Cluster score": score}


class Wasserstein:

    def __init__(self):
        super().__init__()

    def evaluate(
        self,
        rd: pd.DataFrame,
        sd: pd.DataFrame,
    ):
        rd, sd = rd.to_numpy(), sd.to_numpy()
        results = {}
        results["wasserstein"] = (
            (SamplesLoss(loss="sinkhorn")(torch.from_numpy(rd), torch.from_numpy(sd)))
            .numpy()
            .item()
        )
        return results


class PRDC:

    def __init__(self, k: int = 5, metric="euclidean"):
        super().__init__()
        self.k = k
        self.metric = metric

    def evaluate(self, rd: pd.DataFrame, sd: pd.DataFrame):
        rd, sd = rd.to_numpy(), sd.to_numpy()
        rd_distances = self._compute_nearest_neighbour_distances(rd, self.k)
        sd_distances = self._compute_nearest_neighbour_distances(sd, self.k)
        rd_sd_distances = metrics.pairwise_distances(rd, sd, metric=self.metric)

        precision = (
            (rd_sd_distances < np.expand_dims(rd_distances, axis=1)).any(axis=0).mean()
        )

        recall = (
            (rd_sd_distances < np.expand_dims(sd_distances, axis=0)).any(axis=1).mean()
        )

        density = (1.0 / float(self.k)) * (
            rd_sd_distances < np.expand_dims(rd_distances, axis=1)
        ).sum(axis=0).mean()

        coverage = (rd_sd_distances.min(axis=1) < rd_distances).mean()

        return dict(
            precision=precision, recall=recall, density=density, coverage=coverage
        )

    def _get_kth_value(self, unsorted: np.ndarray, k: int, axis: int = -1):
        indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
        k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
        kth_values = k_smallests.max(axis=axis)
        return kth_values

    def _compute_nearest_neighbour_distances(
        self, input_features: np.ndarray, nearest_k: int
    ):
        distances = metrics.pairwise_distances(input_features, input_features)
        radii = self._get_kth_value(distances, k=nearest_k + 1, axis=-1)
        return radii


class MMD:

    def __init__(self, kernel: str = "rbf"):
        super().__init__()
        self.kernel = kernel

    def evaluate(
        self,
        rd: pd.DataFrame,
        sd: pd.DataFrame,
    ):
        if self.kernel == "linear":
            """
            MMD using linear kernel (i.e., k(x,y) = <x,y>)
            """
            delta_df = rd.mean(axis=0) - sd.mean(axis=0)
            delta = delta_df.values

            score = delta.dot(delta.T)
        elif self.kernel == "rbf":
            """
            MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
            """
            gamma = 1.0
            XX = metrics.pairwise.rbf_kernel(
                rd.to_numpy().reshape(len(rd), -1),
                rd.to_numpy().reshape(len(rd), -1),
                gamma,
            )
            YY = metrics.pairwise.rbf_kernel(
                sd.to_numpy().reshape(len(sd), -1),
                sd.to_numpy().reshape(len(sd), -1),
                gamma,
            )
            XY = metrics.pairwise.rbf_kernel(
                rd.to_numpy().reshape(len(rd), -1),
                sd.to_numpy().reshape(len(sd), -1),
                gamma,
            )
            score = XX.mean() + YY.mean() - 2 * XY.mean()
        elif self.kernel == "polynomial":
            """
            MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
            """
            degree = 2
            gamma = 1
            coef0 = 0
            XX = metrics.pairwise.polynomial_kernel(
                rd.to_numpy().reshape(len(rd), -1),
                rd.to_numpy().reshape(len(rd), -1),
                degree,
                gamma,
                coef0,
            )
            YY = metrics.pairwise.polynomial_kernel(
                sd.to_numpy().reshape(len(sd), -1),
                sd.to_numpy().reshape(len(sd), -1),
                degree,
                gamma,
                coef0,
            )
            XY = metrics.pairwise.polynomial_kernel(
                rd.to_numpy().reshape(len(rd), -1),
                sd.to_numpy().reshape(len(sd), -1),
                degree,
                gamma,
                coef0,
            )
            score = XX.mean() + YY.mean() - 2 * XY.mean()
        else:
            raise ValueError(f"Unsupported kernel {self.kernel}")

        return {"MMD": float(score)}
