import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, RocCurveDisplay, auc, precision_recall_curve, PrecisionRecallDisplay


def plot_roc(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Taken straight out of sklearn's guide
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()
    plt.show()


def plot_pr(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Taken straight out of sklearn's guide
    prec, recall, _ = precision_recall_curve(y_test, y_pred, pos_label=1)
    _ = PrecisionRecallDisplay(precision=prec, recall=recall).plot()


def model_binning_summary(model, X, y):
    percentiles = {k: [] for k in range(10)}

    y_pred = model.predict(X)

    for iy, iy_pred in zip(y, y_pred):
        iperc = int(10 * iy_pred)
        percentiles[iperc].append(iy)

    cumulative_n, cumulative_delinquent = 0, 0

    print(f"{'bkt':4s} {'num':5s} {'num_delinquent':12s} "
          f"{'perc_delinquent':15s} {'cumulative_delinquent':20s}")
    for ipercentile, ires in percentiles.items():
        n = len(ires)
        n_delinquent = sum(ires)

        cumulative_n += n
        cumulative_delinquent += n_delinquent
        cumulative_delinquent_rate = cumulative_delinquent / cumulative_n

        print(
            f"{ipercentile / 10:4.2f} {n:5d} {n_delinquent:12d} {100 * n_delinquent / n:15.2f} "
            f"{100 * cumulative_delinquent_rate:20.2f}")
