import copy


def null_transform(df):
    df = copy.deepcopy(df)

    vars_to_exclude = {'id'}

    try:
        id_col = df['id']
        df = df.drop(columns=vars_to_exclude)

    except Exception as e:
        id_col = None
        vars_to_exclude.remove('id')
        df = df.drop(columns=vars_to_exclude)

    return df, id_col
