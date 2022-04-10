import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def summarize_df(df):
    total_rows = len(df)

    # Get N/A counts
    na_counts = df.isna().sum()
    na_counts.name = "N/A count"

    # Get N/A ratio
    na_ratio = na_counts / total_rows
    na_ratio.name = "N/A fraction"

    # Combine counts & ratio into a summary table
    na_df = pd.concat([na_counts, na_ratio], axis=1)
    na_df.index.name = 'Variable Name'
    na_df = na_df.sort_values(by=['N/A count'], ascending=False, kind='heapsort')

    return na_df


def plot_df(df):
    fig = plt.figure(figsize=(25, 20))

    i = 0
    for icol in df.columns:
        if icol in ('SeriousDlqin2yrs', 'id'):
            continue

        plt.subplot(3, 4, i + 1)
        i += 1

        sns.histplot(df, x=icol, hue="SeriousDlqin2yrs", element="step", stat="density", common_norm=False)

    fig.suptitle('Data Column Histograms')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.show()
