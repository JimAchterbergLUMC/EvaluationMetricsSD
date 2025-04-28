import os
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from sklearn.decomposition import PCA
from torchvision import transforms as transforms
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
)
from mlxtend.frequent_patterns import apriori, association_rules
import seaborn as sns
from umap import UMAP
from sklearn.manifold import TSNE

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


class DomainConstraint:

    def __init__(self, constraint_list: list):
        super().__init__()
        self.constraint_list = constraint_list

    def evaluate(self, rd: pd.DataFrame, sd: pd.DataFrame):
        result = {}
        for constraint in self.constraint_list:
            result[constraint] = {}
            result[constraint]["Real"] = rd.eval(constraint).mean()
            result[constraint]["Synthetic"] = sd.eval(constraint).mean()

        return result


class FeatureWisePlots:

    def __init__(
        self,
        discrete_features: list,
        save_dir: str = None,
        plot_cols: int = 3,
        figsize: tuple = (10, 10),
        single_fig: bool = True,
    ):
        super().__init__()
        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        self.discrete_features = discrete_features
        self.plot_cols = plot_cols
        self.figsize = figsize
        self.single_fig = single_fig

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
        np.delete(axes, list(range(i + 1, len(axes))))

        if self.single_fig:
            legend = axes[-1].legend()
            axes[-1].get_legend().remove()

            fig.legend(
                legend.legend_handles,
                [t.get_text() for t in legend.get_texts()],
                loc="lower right",
                fontsize=12,
            )

            plt.tight_layout()

            if self.save_dir is not None:
                plt.savefig(f"{self.save_dir}/all_features.png")

            return plt

        else:

            for i, ax in enumerate(axes):
                fig, new_ax = plt.subplots(figsize=self.figsize)

                # Copy lines
                for line in ax.lines:
                    new_ax.plot(
                        *line.get_data(),
                        label=line.get_label(),
                        color=line.get_color(),
                        linestyle=line.get_linestyle(),
                        linewidth=line.get_linewidth(),
                        marker=line.get_marker(),
                    )

                # Copy scatter plots (collections)
                for collection in ax.collections:
                    offsets = collection.get_offsets()
                    if len(offsets) > 0:
                        x, y = offsets[:, 0], offsets[:, 1]
                        new_ax.scatter(
                            x,
                            y,
                            label=collection.get_label(),
                            color=collection.get_facecolor()[0],
                            marker=collection.get_paths()[0],
                            alpha=collection.get_alpha(),
                        )

                # Copy bars, histograms, patches (e.g., rectangles)
                for patch in ax.patches:
                    new_patch = patch.__class__(
                        xy=(
                            patch.get_xy()
                            if hasattr(patch, "get_xy")
                            else (patch.get_x(), patch.get_y())
                        ),
                        width=patch.get_width(),
                        height=patch.get_height(),
                        angle=getattr(patch, "angle", 0),
                        facecolor=patch.get_facecolor(),
                        edgecolor=patch.get_edgecolor(),
                        alpha=patch.get_alpha(),
                        linewidth=patch.get_linewidth(),
                        linestyle=patch.get_linestyle(),
                    )
                    new_ax.add_patch(new_patch)

                # Copy images (e.g., heatmaps)
                for im in ax.images:
                    new_ax.imshow(
                        im.get_array(),
                        extent=im.get_extent(),
                        origin=im.origin,
                        cmap=im.get_cmap(),
                        alpha=im.get_alpha(),
                        interpolation=im.get_interpolation(),
                    )

                # Copy annotations/text
                for text in ax.texts:
                    new_ax.text(
                        text.get_position()[0],
                        text.get_position()[1],
                        text.get_text(),
                        fontsize=text.get_fontsize(),
                        color=text.get_color(),
                        verticalalignment="top",
                        transform=new_ax.transAxes,
                    )

                # Copy title and labels
                new_ax.set_title(ax.get_title())
                new_ax.set_xlabel(ax.get_xlabel())
                new_ax.set_ylabel(ax.get_ylabel())

                # Copy limits
                new_ax.set_xlim(ax.get_xlim())
                new_ax.set_ylim(ax.get_ylim())

                # Copy legend
                if ax.get_legend():
                    new_ax.legend()

                # Save and close
                if self.save_dir is not None:

                    fig.savefig(
                        f"{self.save_dir}/{ax.get_title()}.png",
                        bbox_inches="tight",
                    )

                plt.close(fig)

            return axes


class CorrelationMatrices:
    def __init__(
        self,
        discrete_features: list,
        save_dir: str = None,
        figsize: tuple = (10, 10),
        single_fig: bool = True,
    ):
        """
        Compute correlation matrices of SD and RD.
        Saves correlation matrices as figures in save_dir if save_dir is not None.
        Else, returns the correlation matrices for further processing.
        """
        super().__init__()
        self.discrete_features = discrete_features
        self.figsize = figsize
        self.single_fig = single_fig
        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def evaluate(self, rd: pd.DataFrame, sd: pd.DataFrame):
        self.numerical_features = [
            x for x in rd.columns if x not in self.discrete_features
        ]

        corr_rd = self.compute_mixed_correlation_matrix(rd)
        corr_sd = self.compute_mixed_correlation_matrix(sd)

        if self.save_dir is None:
            return {"Correlation RD": corr_rd, "Correlation SD": corr_sd}

        if self.single_fig:
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
            plt.savefig(f"{self.save_dir}/correlation_all.png")
        else:
            fig1, axs = plt.subplots(figsize=self.figsize)
            sns.heatmap(
                corr_rd,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                center=0,
                linewidths=0.5,
                ax=axs,
                square=True,
                cbar_kws={"shrink": 0.5},
                vmin=-1,
                vmax=1,
            )
            axs.set_title("Real")
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/correlation_rd.png")
            fig2, axs = plt.subplots(figsize=self.figsize)
            sns.heatmap(
                corr_sd,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                center=0,
                linewidths=0.5,
                square=True,
                ax=axs,
                cbar_kws={"shrink": 0.5},
                vmin=-1,
                vmax=1,
            )
            axs.set_title("Synthetic")
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/correlation_sd.png")

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
            f"ARM #Real rules (bins={self.n_bins},support={self.min_support},confidence={self.min_confidence})": len(
                rd_rules
            ),
            "ARM #Synthetic rules (bins={self.n_bins},support={self.min_support},confidence={self.min_confidence})": len(
                sd_rules
            ),
            "ARM Precision (bins={self.n_bins},support={self.min_support},confidence={self.min_confidence})": precision,
            "ARM Recall (bins={self.n_bins},support={self.min_support},confidence={self.min_confidence})": recall,
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
        model_name: str = "xgb",
        metric_numerical: str = "rmse",
        metric_discrete: str = "roc_auc",
        k: int = 3,
    ):
        super().__init__()
        self.model_name = model_name
        self.metric_numerical = metric_numerical
        self.metric_discrete = metric_discrete
        self.k = k

        # if not reduced should return something like {DWP {k} {XGB} {Real/Synthetic} {Attribute} {AUC/RMSE}: score}
        # else if reduced by discrepancy: {DWP {k} {XGB} {Attribute} {AUC/RMSE}: score_discrepancy}
        # else if reduced by avg. discrepancy: {DWP {k} {XGB} {AUC/RMSE}: avg. score_discrepancy}

    def evaluate(self, rd: pd.DataFrame, sd: pd.DataFrame):

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
            pass

        scores_sd = {col: [] for col in x_rd.columns}
        scores_rd = {col: [] for col in x_rd.columns}
        cv = KFold(n_splits=self.k)
        for idx_tr, idx_te in cv.split(rd):
            rd_train, rd_test = x_rd.iloc[idx_tr], x_rd.iloc[idx_te]
            sd_train = x_sd.iloc[idx_tr]
            for col in rd_train.columns:
                # select appropriate model and performance metric
                if col in self.discretes:
                    model = CLF[self.model_name]
                    metric = ACCURACY_METRICS[self.metric_discrete]
                else:
                    model = REG[self.model_name]
                    metric = ACCURACY_METRICS[self.metric_numerical]

                # fit on sd and test on rd
                model.fit(sd_train.drop(col, axis=1), sd_train[col])
                preds_sd = self._predict(sd_train, rd_test, col, model)
                score_sd = metric(rd_test[col], preds_sd)
                scores_sd[col].append(score_sd)

                # fit and test on rd
                model.fit(rd_train.drop(col, axis=1), rd_train[col])
                preds_rd = self._predict(rd_train, rd_test, col, model)
                score_rd = metric(rd_test[col], preds_rd)
                scores_rd[col].append(score_rd)

        avg_scores_sd = {
            col: sum(scores) / len(scores) for col, scores in scores_sd.items()
        }
        avg_scores_rd = {
            col: sum(scores) / len(scores) for col, scores in scores_rd.items()
        }

        return {"DWP Real": avg_scores_rd, "DWP Synthetic": avg_scores_sd}

    def _predict(self, train, test, target, model):
        if target in self.discretes and self.metric_discrete == "roc_auc":
            if len(np.unique(train[target])) == 2:
                return model.predict_proba(test.drop(target, axis=1))[:, 1]
            else:
                return model.predict_proba(test.drop(target, axis=1))
        else:
            return model.predict(test.drop(target, axis=1))

    def dwp(self, train: pd.DataFrame, test: pd.DataFrame):
        dwp = {}
        for col in train.columns:
            if col in self.discrete_features:
                model = CLF[self.model_name]
                metric = ACCURACY_METRICS[self.metric_discrete]
            else:
                model = REG[self.model_name]
                metric = ACCURACY_METRICS[self.metric_numerical]

            X_tr, y_tr, X_te, y_te = preprocess_prediction(
                train, test, col, self.discrete_features
            )
            model.fit(X_tr, y_tr)

            if col in self.discrete_features:
                if self.metric_discrete == "roc_auc":
                    preds = model.predict_proba(X_te)
                    if len(np.unique(y_tr)) == 2:
                        preds = preds[:, 1]
                else:
                    preds = model.predict(X_te)
            else:
                preds = model.predict(X_te)
            score = metric(y_te, preds)
            dwp[col] = score
        return dwp


class Projections:

    def __init__(
        self,
        embedder: str = "pca",
        save_dir: str = None,
        figsize: tuple = (10, 10),
        plot: str = "scatter",
        **embedder_kwargs: dict,
    ):
        super().__init__()
        self.figsize = figsize
        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        self.embedder_name = embedder
        EMBEDDERS = {"pca": PCA, "umap": UMAP, "tsne": TSNE}
        self.embedder = EMBEDDERS[embedder.lower()]
        self.embedder_kwargs = embedder_kwargs
        self.plot = plot

    def evaluate(
        self,
        rd: pd.DataFrame,
        sd: pd.DataFrame,
    ):
        if self.plot == "scatter":
            self.embedder_kwargs["n_components"] = 2
        else:
            if "n_components" not in self.embedder_kwargs.keys():
                raise Exception(
                    "Please input n_components for Projections as embedder_kwargs when not using scatterplot, so the metric knows how many components to plot."
                )

        data = pd.concat([rd, sd])
        self.embedder = self.embedder(**self.embedder_kwargs)
        emb = self.embedder.fit_transform(data)
        rd_emb = emb[: len(rd)]
        sd_emb = emb[len(rd) :]

        if self.plot == "scatter":
            # scatterplot of first two components
            fig, ax = plt.subplots(figsize=self.figsize)
            sns.scatterplot(
                x=rd_emb[:, 0],
                y=rd_emb[:, 1],
                color="blue",
                ax=ax,
                label="Real",
            )
            sns.scatterplot(
                x=sd_emb[:, 0],
                y=sd_emb[:, 1],
                color="red",
                ax=ax,
                label="Synthetic",
            )
        elif self.plot == "marginal":
            n_cols = min(3, rd_emb.shape[1])
            n_rows = int(np.ceil(rd_emb.shape[1] / n_cols))
            fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=self.figsize)
            axs = axs.flatten()
            for i in range(rd_emb.shape[1]):
                sns.kdeplot(rd_emb[:, i], ax=axs[i], color="blue", alpha=0.3, fill=True)
                sns.kdeplot(sd_emb[:, i], ax=axs[i], color="red", alpha=0.3, fill=True)
                axs[i].set_title(f"Component {i+1}")
                axs[i].set_xlabel("")
                axs[i].set_ylabel("")
            for ax in axs[rd_emb.shape[1] :]:
                ax.axis("off")

        fig.suptitle(self.embedder_name.upper())
        plt.tight_layout()

        if self.save_dir is not None:
            plt.savefig(f"{self.save_dir}/{self.embedder_name}.png")


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
            (
                SamplesLoss(loss="sinkhorn")(
                    torch.from_numpy(rd).contiguous(), torch.from_numpy(sd).contiguous()
                )
            )
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

        return {
            f"precision (k={self.k})": precision,
            f"recall (k={self.k})": recall,
            f"density (k={self.k})": density,
            f"coverage (k={self.k})": coverage,
        }

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

        return {f"MMD ({self.kernel})": float(score)}


class ClassifierTest:

    # TBD: add support for other classifiers than XGB

    def __init__(self, clf: str = "xgb", kfolds: int = 3, random_state: int = 0):
        super().__init__()
        self.clf = clf
        self.kfolds = kfolds
        self.random_state = random_state

    def evaluate(
        self,
        rd: pd.DataFrame,
        sd: pd.DataFrame,
    ):
        """
        Expects non-preprocessed data.
        """

        if self.clf == "xgb":
            # for xgb classifier use built-in functionality to handle categorical data
            X = pd.DataFrame()
            for col in rd.columns:
                try:
                    rd[col].astype(float)
                    # align numerical precision between sd and rd to avoid XGB splitting based on precision discrepancies
                    sd[col], rd[col] = self.align_column_precision(sd[col], rd[col])
                    X[col] = pd.concat([rd[col], sd[col]])
                    X[col] = X[col].astype("float")
                except:
                    X[col] = pd.concat([rd[col], sd[col]])
                    X[col] = X[col].astype("category")

            # add XGB support for categorical dtypes
            model = CLF[self.clf]
            model.set_params(tree_method="hist", enable_categorical=True)

        else:
            # TBD: add functionality for other classifiers
            pass

        y = np.concatenate((np.zeros(len(rd)), np.ones(len(sd))))
        y = pd.Series(y, name="y")
        X = X.reset_index(drop=True)
        # X = X.to_numpy()

        skf = StratifiedKFold(
            n_splits=self.kfolds, shuffle=True, random_state=self.random_state
        )
        scores = []
        for idx_tr, idx_te in skf.split(X, y):
            # TBD: add functionality for other classifiers
            X_tr, X_te = X.iloc[idx_tr, :], X.iloc[idx_te, :]
            y_tr, y_te = y[idx_tr], y[idx_te]
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_te)
            scores.append(roc_auc_score(y_te, preds[:, 1]))

        return {
            f"Classifier test AUC ({self.kfolds} folds, {self.clf} classifier)": np.mean(
                scores
            )
        }

    def align_column_precision(
        self, target_series: pd.Series, reference_series: pd.Series
    ):
        """
        Aligns target series precision according to the meaningful precision of the reference series.
        Reference series also gets rounded to its meaningful precision.
        Meaningful precision means most granular precision without trailing zeros.
        """

        def get_max_decimal_places(series):
            def count_decimal_places(x):
                try:
                    s = (
                        format(x, ".16f").rstrip("0").rstrip(".")
                    )  # Full precision float, no trailing zeros
                    if "." in s:
                        return len(s.split(".")[-1])
                    return 0
                except:
                    return 0  # In case of NaNs or errors

            return series.dropna().apply(count_decimal_places).max()

        # Find max meaningful precision across both
        precision = get_max_decimal_places(reference_series)
        target_series = target_series.round(precision)
        reference_series = reference_series.round(precision)

        target_series, reference_series = target_series.astype(
            float
        ), reference_series.astype(float)

        return target_series, reference_series
