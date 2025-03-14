# =====================#
# ---- Libraries ---- #
# =====================#

import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

# =========================#
# ---- Main Function ---- #
# =========================#


def process_data(
    X,
    categorical_features=None,
    label=None,
    training=True,
    encoder=None,
    lb=None
):
    """Processes the data used in the machine learning pipeline.

    This function applies one-hot encoding to categorical features and label
    binarization for labels. It can be used in both training and
    inference/validation.

    Note: Depending on the type of model used,
    you may want to add functionality
    that scales continuous data.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing the features and label.
        Columns in `categorical_features`
    categorical_features : list[str], optional
        List containing the names of
        the categorical features (default=None)
    label : str, optional
        Name of the label column in `X`.
        If None, then an empty array will be returned
        for `y` (default=None)
    training : bool, optional
        Indicates whether the function is
        being used for training or
        inference/validation.
    encoder : OneHotEncoder, optional
        Pre-trained sklearn OneHotEncoder,
        only used if `training=False`.
    lb : LabelBinarizer, optional
        Pre-trained sklearn LabelBinarizer,
        only used if `training=False`.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if `label` is provided,
        otherwise an empty np.array.
    encoder : OneHotEncoder
        Trained OneHotEncoder if `training=True`,
        otherwise returns the provided encoder.
    lb : LabelBinarizer
        Trained LabelBinarizer\
            if `training=True`,
        otherwise returns the provided binarizer.
    """

    if categorical_features is None:
        categorical_features = []

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(columns=categorical_features, axis=1)

    if training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
