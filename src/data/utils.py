import pandas as pd


def read_df(path):
    raw_df = pd.read_csv(path)
    raw_df.rename(columns={raw_df.columns.values[0]: 'id'}, inplace=True)

    return raw_df


def get_xy(df):
    X = df.drop(columns=['SeriousDlqin2yrs'])
    y = df['SeriousDlqin2yrs']

    return X, y
