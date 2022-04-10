import pandas as pd

from sklearn.metrics import roc_auc_score
from src.data.transforms.null_transform import null_transform
from src.data.utils import read_df, get_xy


def eval_auc(model, X_test, y_test):
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)

    return auc


def generate_test_output(input_csv, model, output_csv):
    test_df = read_df(input_csv)
    _, tdf_id = null_transform(test_df)

    tX, ty = get_xy(test_df)
    typred = model.predict(tX)
    typred = pd.Series(typred)

    typred.name = 'Probability'
    pred_out = pd.concat([tdf_id, typred], axis=1)
    pred_out.to_csv(output_csv, index=False)

    return pred_out
