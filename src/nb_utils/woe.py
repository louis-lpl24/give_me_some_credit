from optbinning import OptimalBinning


def summarize_woe_iv(df, idcol, ycol):
    splits = {}

    xcols = set(df.columns) - {ycol, idcol}

    y = df[ycol]

    for icol in xcols:
        print(icol)
        ix = df[icol]

        optb = OptimalBinning(name=icol, dtype="numerical", solver="cp")
        optb.fit(ix, y)
        assert optb.status == 'OPTIMAL'

        binning_table = optb.binning_table
        binning_df = binning_table.build()

        display(binning_df)
        binning_table.plot(metric="woe")

        splits[icol] = optb

    return splits
