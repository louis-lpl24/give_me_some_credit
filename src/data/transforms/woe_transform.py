import copy
import joblib

from optbinning import OptimalBinning
from src.nb_utils.misc import is_notebook


def forced_binning(df, xcol, ycol, monotonic_trend, gamma, user_splits=None, silent=False):
    if silent is False:
        print(f"x={xcol}, y={ycol}")

    x = df[xcol]
    y = df[ycol]

    optb = OptimalBinning(name=xcol, dtype="numerical", solver="cp", monotonic_trend=monotonic_trend,
                          gamma=gamma, user_splits=user_splits)
    optb.fit(x, y)
    assert optb.status == 'OPTIMAL'

    binning_table = optb.binning_table
    binning_df = binning_table.build()

    if silent is False:
        if is_notebook() is True:
            display(binning_df)
            binning_table.plot(metric="woe")

    return optb


class WoETransform(object):
    def __init__(self):
        self.NumberOfDependentsBins = None
        self.DebtRatioBins = None
        self.MonthlyIncomeBins = None
        self.ageBins = None
        self.RevolvingUtilizationOfUnsecuredLinesBins = None
        self.NumberOfOpenCreditLinesAndLoansBins = None
        self.NumberRealEstateLoansOrLinesBins = None
        self.NumberOfTime3059DaysPastDueNotWorseBins = None
        self.inited = False
        self.vars_to_exclude = [
            'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfTimes90DaysLate'
        ]

    def woe_fit(self, df):
        self.NumberOfDependentsBins = forced_binning(
            df, 'NumberOfDependents', 'SeriousDlqin2yrs',
            'ascending', 0, silent=True)

        self.DebtRatioBins = forced_binning(
            df, 'DebtRatio', 'SeriousDlqin2yrs',
            'ascending', 0, silent=True)

        self.MonthlyIncomeBins = forced_binning(
            df, 'MonthlyIncome', 'SeriousDlqin2yrs',
            'descending', 0, silent=True)

        self.ageBins = forced_binning(
            df, 'age', 'SeriousDlqin2yrs',
            'descending', 0.1, silent=True)

        self.RevolvingUtilizationOfUnsecuredLinesBins = forced_binning(
            df, 'RevolvingUtilizationOfUnsecuredLines', 'SeriousDlqin2yrs',
            'ascending', 0.2, silent=True)

        self.NumberOfOpenCreditLinesAndLoansBins = forced_binning(
            df, 'NumberOfOpenCreditLinesAndLoans', 'SeriousDlqin2yrs',
            'descending', 0, silent=True)

        self.NumberRealEstateLoansOrLinesBins = forced_binning(
            df, 'NumberRealEstateLoansOrLines', 'SeriousDlqin2yrs',
            None, 0, user_splits=[1, 2, 3], silent=True)

        self.NumberOfTime3059DaysPastDueNotWorseBins = forced_binning(
            df, 'NumberOfTime30-59DaysPastDueNotWorse', 'SeriousDlqin2yrs',
            None, 0, user_splits=[0, 1, 2], silent=True)

        self.inited = True

    def save(self, path):
        assert self.inited is True

        save_dict = {
            'NumberOfDependentsBins': self.NumberOfDependentsBins,
            'DebtRatioBins': self.DebtRatioBins,
            'MonthlyIncomeBins': self.MonthlyIncomeBins,
            'ageBins': self.ageBins,
            'RevolvingUtilizationOfUnsecuredLinesBins': self.RevolvingUtilizationOfUnsecuredLinesBins,
            'NumberOfOpenCreditLinesAndLoansBins': self.NumberOfOpenCreditLinesAndLoansBins,
            'NumberRealEstateLoansOrLinesBins': self.NumberRealEstateLoansOrLinesBins,
            'NumberOfTime3059DaysPastDueNotWorseBins': self.NumberOfTime3059DaysPastDueNotWorseBins
        }

        joblib.dump(save_dict, path)
        print(f"Saved to {path}")

    def load(self, path):
        load_dict = joblib.load(path)

        self.NumberOfDependentsBins = load_dict['NumberOfDependentsBins']
        self.DebtRatioBins = load_dict['DebtRatioBins']
        self.MonthlyIncomeBins = load_dict['MonthlyIncomeBins']
        self.ageBins = load_dict['ageBins']
        self.RevolvingUtilizationOfUnsecuredLinesBins = load_dict['RevolvingUtilizationOfUnsecuredLinesBins']
        self.NumberOfOpenCreditLinesAndLoansBins = load_dict['NumberOfOpenCreditLinesAndLoansBins']
        self.NumberRealEstateLoansOrLinesBins = load_dict['NumberRealEstateLoansOrLinesBins']
        self.NumberOfTime3059DaysPastDueNotWorseBins = load_dict['NumberOfTime3059DaysPastDueNotWorseBins']

        self.inited = True

    def __call__(self, df):
        assert self.inited is True

        df = copy.deepcopy(df)

        id_col = df['id']
        df = df.drop(columns=self.vars_to_exclude)

        df['NumberOfDependents'] = self.NumberOfDependentsBins.transform(df['NumberOfDependents'], metric="woe")
        df['DebtRatio'] = self.DebtRatioBins.transform(df['DebtRatio'], metric="woe")
        df['MonthlyIncome'] = self.MonthlyIncomeBins.transform(df['MonthlyIncome'], metric="woe")
        df['age'] = self.ageBins.transform(df['age'], metric="woe")
        df['RevolvingUtilizationOfUnsecuredLines'] = self.RevolvingUtilizationOfUnsecuredLinesBins.transform(
            df['RevolvingUtilizationOfUnsecuredLines'], metric="woe")
        df['NumberOfOpenCreditLinesAndLoans'] = self.NumberOfOpenCreditLinesAndLoansBins.transform(
            df['NumberOfOpenCreditLinesAndLoans'], metric="woe")
        df['NumberRealEstateLoansOrLines'] = self.NumberRealEstateLoansOrLinesBins.transform(
            df['NumberRealEstateLoansOrLines'], metric="woe")
        df['NumberOfTime30-59DaysPastDueNotWorse'] = self.NumberOfTime3059DaysPastDueNotWorseBins.transform(
            df['NumberOfTime30-59DaysPastDueNotWorse'], metric="woe")

        return df, id_col


class WoETransformV2(object):
    def __init__(self):
        self.NumberOfDependentsBins = None
        self.DebtRatioBins = None
        self.MonthlyIncomeBins = None
        self.ageBins = None
        self.RevolvingUtilizationOfUnsecuredLinesBins = None
        self.NumberOfOpenCreditLinesAndLoansBins = None
        self.NumberRealEstateLoansOrLinesBins = None
        self.NumberOfTime3059DaysPastDueNotWorseBins = None
        self.NumberOfTime6089DaysPastDueNotWorseBins = None
        self.NumberOfTimes90DaysLateBins = None
        self.inited = False

    def woe_fit(self, df):
        self.NumberOfDependentsBins = forced_binning(
            df, 'NumberOfDependents', 'SeriousDlqin2yrs',
            'ascending', 0, silent=True)

        self.DebtRatioBins = forced_binning(
            df, 'DebtRatio', 'SeriousDlqin2yrs',
            'ascending', 0, silent=True)

        self.MonthlyIncomeBins = forced_binning(
            df, 'MonthlyIncome', 'SeriousDlqin2yrs',
            'descending', 0, silent=True)

        self.ageBins = forced_binning(
            df, 'age', 'SeriousDlqin2yrs',
            'descending', 0.1, silent=True)

        self.RevolvingUtilizationOfUnsecuredLinesBins = forced_binning(
            df, 'RevolvingUtilizationOfUnsecuredLines', 'SeriousDlqin2yrs',
            'ascending', 0.2, silent=True)

        self.NumberOfOpenCreditLinesAndLoansBins = forced_binning(
            df, 'NumberOfOpenCreditLinesAndLoans', 'SeriousDlqin2yrs',
            'descending', 0, silent=True)

        self.NumberRealEstateLoansOrLinesBins = forced_binning(
            df, 'NumberRealEstateLoansOrLines', 'SeriousDlqin2yrs',
            None, 0, user_splits=[1, 2, 3], silent=True)

        self.NumberOfTime3059DaysPastDueNotWorseBins = forced_binning(
            df, 'NumberOfTime30-59DaysPastDueNotWorse', 'SeriousDlqin2yrs',
            None, 0, user_splits=[0, 1, 2], silent=True)

        self.NumberOfTime6089DaysPastDueNotWorseBins = forced_binning(
            df, 'NumberOfTime60-89DaysPastDueNotWorse', 'SeriousDlqin2yrs',
            None, 0, user_splits=[0, 1, 2], silent=True)

        self.NumberOfTimes90DaysLateBins = forced_binning(
            df, 'NumberOfTimes90DaysLate', 'SeriousDlqin2yrs',
            None, 0, user_splits=[0, 1, 2], silent=True)

        self.inited = True

    def save(self, path):
        assert self.inited is True

        save_dict = {
            'NumberOfDependentsBins': self.NumberOfDependentsBins,
            'DebtRatioBins': self.DebtRatioBins,
            'MonthlyIncomeBins': self.MonthlyIncomeBins,
            'ageBins': self.ageBins,
            'RevolvingUtilizationOfUnsecuredLinesBins': self.RevolvingUtilizationOfUnsecuredLinesBins,
            'NumberOfOpenCreditLinesAndLoansBins': self.NumberOfOpenCreditLinesAndLoansBins,
            'NumberRealEstateLoansOrLinesBins': self.NumberRealEstateLoansOrLinesBins,
            'NumberOfTime3059DaysPastDueNotWorseBins': self.NumberOfTime3059DaysPastDueNotWorseBins,
            'NumberOfTime6089DaysPastDueNotWorseBins': self.NumberOfTime6089DaysPastDueNotWorseBins,
            'NumberOfTimes90DaysLateBins': self.NumberOfTimes90DaysLateBins
        }

        joblib.dump(save_dict, path)
        print(f"Saved to {path}")

    def load(self, path):
        load_dict = joblib.load(path)

        self.NumberOfDependentsBins = load_dict['NumberOfDependentsBins']
        self.DebtRatioBins = load_dict['DebtRatioBins']
        self.MonthlyIncomeBins = load_dict['MonthlyIncomeBins']
        self.ageBins = load_dict['ageBins']
        self.RevolvingUtilizationOfUnsecuredLinesBins = load_dict['RevolvingUtilizationOfUnsecuredLinesBins']
        self.NumberOfOpenCreditLinesAndLoansBins = load_dict['NumberOfOpenCreditLinesAndLoansBins']
        self.NumberRealEstateLoansOrLinesBins = load_dict['NumberRealEstateLoansOrLinesBins']
        self.NumberOfTime3059DaysPastDueNotWorseBins = load_dict['NumberOfTime3059DaysPastDueNotWorseBins']
        self.NumberOfTime6089DaysPastDueNotWorseBins = load_dict['NumberOfTime6089DaysPastDueNotWorseBins']
        self.NumberOfTimes90DaysLateBins = load_dict['NumberOfTimes90DaysLateBins']

        self.inited = True

    def __call__(self, df):
        assert self.inited is True

        df = copy.deepcopy(df)

        id_col = df['id']

        df['NumberOfDependents'] = self.NumberOfDependentsBins.transform(df['NumberOfDependents'], metric="woe")
        df['DebtRatio'] = self.DebtRatioBins.transform(df['DebtRatio'], metric="woe")
        df['MonthlyIncome'] = self.MonthlyIncomeBins.transform(df['MonthlyIncome'], metric="woe")
        df['age'] = self.ageBins.transform(df['age'], metric="woe")
        df['RevolvingUtilizationOfUnsecuredLines'] = self.RevolvingUtilizationOfUnsecuredLinesBins.transform(
            df['RevolvingUtilizationOfUnsecuredLines'], metric="woe")
        df['NumberOfOpenCreditLinesAndLoans'] = self.NumberOfOpenCreditLinesAndLoansBins.transform(
            df['NumberOfOpenCreditLinesAndLoans'], metric="woe")
        df['NumberRealEstateLoansOrLines'] = self.NumberRealEstateLoansOrLinesBins.transform(
            df['NumberRealEstateLoansOrLines'], metric="woe")
        df['NumberOfTime30-59DaysPastDueNotWorse'] = self.NumberOfTime3059DaysPastDueNotWorseBins.transform(
            df['NumberOfTime30-59DaysPastDueNotWorse'], metric="woe")
        df['NumberOfTime60-89DaysPastDueNotWorse'] = self.NumberOfTime6089DaysPastDueNotWorseBins.transform(
            df['NumberOfTime60-89DaysPastDueNotWorse'], metric="woe")
        df['NumberOfTimes90DaysLate'] = self.NumberOfTimes90DaysLateBins.transform(
            df['NumberOfTimes90DaysLate'], metric="woe")

        return df, id_col
