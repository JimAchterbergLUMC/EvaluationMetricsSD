# contains helper functions for preprocessing

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler


def preprocess_eval(
    train: pd.DataFrame,
    test: pd.DataFrame,
    syn: pd.DataFrame,
    ohe_threshold: int = 15,
):
    """
    Perform preprocessing.
    One hot encode low cardinality features, label-encode high cardinality features, and minmax scale s.t. range=[0,1] for all data.
    """
    all_df = pd.concat([train, test, syn], ignore_index=True)
    all_df_ = []
    for col in all_df.columns:
        # one hot encode if nunique below threshold
        if all_df[col].nunique() <= ohe_threshold:
            encoder = OneHotEncoder(sparse_output=False, drop="if_binary")
            data = encoder.fit_transform(all_df[[col]])
            all_df_.append(pd.DataFrame(data, columns=encoder.get_feature_names_out()))
        else:
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
    train = all_df[: len(train)].copy()
    test = all_df[len(train) : len(train) + len(test)].copy()
    syn = all_df[len(train) + len(test) :].copy()

    scaler = MinMaxScaler().fit(train)
    train = pd.DataFrame(
        scaler.transform(train), columns=scaler.get_feature_names_out()
    )
    test = pd.DataFrame(scaler.transform(test), columns=scaler.get_feature_names_out())
    syn = pd.DataFrame(scaler.transform(syn), columns=scaler.get_feature_names_out())
    return train, test, syn


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
