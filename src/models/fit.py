import copy
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split
from src.data.utils import get_xy
from src.utils.eval import eval_auc


def kfold_fit(df, n_splits, model_class, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {}

    X, y = get_xy(df)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    print(f"kfold x-val for k={skf.get_n_splits(X, y)}")

    for irun, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = copy.deepcopy(X.iloc[train_index, :]), copy.deepcopy(X.iloc[test_index, :])
        y_train, y_test = copy.deepcopy(y.iloc[train_index]), copy.deepcopy(y.iloc[test_index])

        lg_model = model_class(**model_kwargs)
        lg_model.fit(X_train, y_train)
        iauc = eval_auc(lg_model, X_test, y_test)

        print(f"run {irun + 1} - class [0 1] for train={np.bincount(y_train)}, "
              f"test={np.bincount(y_test)} --> AUC={iauc:.3f}")

    final_model = model_class()
    final_model.fit(X, y)

    return final_model


def df_train_test_stratify_split(df, test_size=0.3):
    X, y = get_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

    return X_train, X_test, y_train, y_test


def fit_and_eval(model_class, X_train, X_test, y_train, y_test, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {}

    final_model = model_class(**model_kwargs)
    final_model.fit(X_train, y_train)
    iauc = eval_auc(final_model, X_test, y_test)

    print(f"class [0 1] for train={np.bincount(y_train)}, test={np.bincount(y_test)} - auc={iauc:.3f}")

    return final_model
