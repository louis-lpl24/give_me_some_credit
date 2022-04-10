import copy
import numpy as np


def drop_correlated_vars_impute_transform(df):
    df = copy.deepcopy(df)

    vars_to_exclude = {'id', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse'}

    try:
        id_col = df['id']
        df = df.drop(columns=vars_to_exclude)

    except Exception as e:
        id_col = None
        vars_to_exclude.remove('id')
        df = df.drop(columns=vars_to_exclude)

    # Impute as discussed
    df['NumberOfDependents'].replace(np.nan, 0, inplace=True)
    df['MonthlyIncome'].replace(np.nan, 0, inplace=True)

    return df, id_col


def log_balance_vars_transform(df):
    df, id_col = drop_correlated_vars_impute_transform(df)

    loggable = {
        "MonthlyIncome",
        "DebtRatio",
        "RevolvingUtilizationOfUnsecuredLines"
    }

    # Fancy fn that can take the log of a data but handles log(0) = inf
    df = df.apply(lambda x: np.sign(x) * np.log(1 + np.abs(x)) if x.name in loggable else x)

    return df, id_col


def log_almost_all_vars_transform(df):
    df, id_col = drop_correlated_vars_impute_transform(df)

    non_loggable = {
        "age",
        "SeriousDlqin2yrs"
    }

    # Fancy fn that can take the log of a data but handles log(0) = inf
    df = df.apply(lambda x: np.sign(x) * np.log(1 + np.abs(x)) if x.name not in non_loggable else x)

    return df, id_col


def log_almost_all_vars_transform_v2(df):
    vars_to_exclude = {'id'}

    try:
        id_col = df['id']
        df = df.drop(columns=vars_to_exclude)

    except Exception as e:
        id_col = None
        vars_to_exclude.remove('id')
        df = df.drop(columns=vars_to_exclude)

    # Impute as discussed
    df['NumberOfDependents'].replace(np.nan, 0, inplace=True)
    df['MonthlyIncome'].replace(np.nan, 0, inplace=True)

    non_loggable = {
        "age",
        "SeriousDlqin2yrs"
    }

    # Fancy fn that can take the log of a data but handles log(0) = inf
    df = df.apply(lambda x: np.sign(x) * np.log(1 + np.abs(x)) if x.name not in non_loggable else x)

    return df, id_col


def log_almost_all_vars_transform_v3(df):
    vars_to_exclude = {'id'}

    try:
        id_col = df['id']
        df = df.drop(columns=vars_to_exclude)

    except Exception as e:
        id_col = None
        vars_to_exclude.remove('id')
        df = df.drop(columns=vars_to_exclude)

    # Impute as discussed
    df['NumberOfDependents'].replace(np.nan, 0, inplace=True)
    df['MonthlyIncome'].replace(np.nan, 0, inplace=True)

    df['MaxPastDue'] = df[['NumberOfTimes90DaysLate',
                           'NumberOfTime60-89DaysPastDueNotWorse',
                           'NumberOfTime30-59DaysPastDueNotWorse']].max(axis=1)

    df = df.drop(columns=['NumberOfTimes90DaysLate',
                          'NumberOfTime60-89DaysPastDueNotWorse',
                          'NumberOfTime30-59DaysPastDueNotWorse'])

    non_loggable = {
        "age",
        "SeriousDlqin2yrs"
    }

    # Fancy fn that can take the log of a data but handles log(0) = inf
    df = df.apply(lambda x: np.sign(x) * np.log(1 + np.abs(x)) if x.name not in non_loggable else x)

    return df, id_col
