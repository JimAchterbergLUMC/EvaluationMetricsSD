# contains helper functions for preprocessing

import pandas as pd
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    MinMaxScaler,
    StandardScaler,
)

SCALERS = {"minmax": MinMaxScaler, "standard": StandardScaler}


def preprocess_prediction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    discrete_features: list = [],
    normalization: str = "standard",
):
    # one hot encode discrete non-target cols
    X = pd.concat([train, test], ignore_index=True)
    ohe_cols = [x for x in discrete_features if x != target_col]
    encoder = OneHotEncoder(sparse_output=False, drop="if_binary")
    X_ohe = pd.DataFrame(
        encoder.fit_transform(X[ohe_cols]),
        columns=encoder.get_feature_names_out(ohe_cols),
    )
    X = X.drop(columns=ohe_cols)
    X = pd.concat(
        [X, X_ohe],
        axis=1,
    )
    X_tr, X_te = X[: len(train)], X[len(train) :]

    # separate X and y
    X_tr, y_tr, X_te, y_te = (
        X_tr.drop(target_col, axis=1),
        X_tr[target_col],
        X_te.drop(target_col, axis=1),
        X_te[target_col],
    )

    # scale numerical features
    numerical_features = [
        x
        for x in X_tr.columns
        if x not in discrete_features
        and x not in encoder.get_feature_names_out(ohe_cols)
    ]
    scaler = SCALERS[normalization]()
    X_tr[numerical_features] = scaler.fit_transform(X_tr[numerical_features])
    X_te[numerical_features] = scaler.transform(X_te[numerical_features])

    if target_col not in discrete_features:
        # normalize numerical targets
        scaler = SCALERS[normalization]()
        y_tr = pd.Series(
            scaler.fit_transform(y_tr.to_frame()).flatten(), name=y_tr.name
        )
        y_te = pd.Series(scaler.transform(y_te.to_frame()).flatten(), name=y_te.name)
    else:
        # label encode discrete targets
        scaler = LabelEncoder()
        y_tr = pd.Series(scaler.fit_transform(y_tr), name=y_tr.name)
        y_te = pd.Series(scaler.transform(y_te), name=y_te.name)

    return X_tr, y_tr, X_te, y_te


def preprocess_eval(
    train: pd.DataFrame,
    test: pd.DataFrame,
    syn: pd.DataFrame,
    ohe_threshold: int = 15,
    normalization: str = "minmax",
    **normalization_kwargs: dict
):
    """
    Perform preprocessing.
    One hot encode low cardinality features, label-encode high cardinality features, and normalize.
    """
    all_df = pd.concat([train, test, syn], ignore_index=True)
    all_df_ = []
    numericals = []
    for col in all_df.columns:
        # one hot encode if nunique below threshold
        if all_df[col].nunique() <= ohe_threshold:
            encoder = OneHotEncoder(sparse_output=False, drop="if_binary")
            data = encoder.fit_transform(all_df[[col]])
            all_df_.append(pd.DataFrame(data, columns=encoder.get_feature_names_out()))
        else:
            numericals.append(col)
            # leave numerical data
            try:
                all_df[col] = all_df[col].astype(float)
                all_df_.append(all_df[col])
            # label encode high cardinality non-numerical data
            except:
                encoder = LabelEncoder()
                data = encoder.fit_transform(all_df[col])
                all_df_.append(pd.Series(data, name=col))

    all_df = pd.concat(all_df_, axis=1)
    tr = all_df[: len(train)].copy()
    te = all_df[len(train) : len(train) + len(test)].copy()
    sd = all_df[len(train) + len(test) :].copy()

    scaler = SCALERS[normalization](**normalization_kwargs)
    scaler.fit(tr[numericals])
    tr[numericals] = scaler.transform(tr[numericals])
    te[numericals] = scaler.transform(te[numericals])
    sd[numericals] = scaler.transform(sd[numericals])
    return tr, te, sd


def determine_feature_types(df: pd.DataFrame, threshold: int = 15):
    numerical = []
    categorical = []

    for col in df.columns:
        try:
            col_data = df[col].astype(float)
            if col_data.nunique() >= threshold:
                numerical.append(col)
            else:
                categorical.append(col)
        except ValueError:
            categorical.append(col)

    return numerical, categorical
