import pandas as pd

# Preprocessor to use with the pipeline
class Preprocessor:
    MEDIAN_INCOME = 51381.5

    def __init__(self):
        self.median_income = self.MEDIAN_INCOME

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['Income'].fillna(self.median_income, inplace=True)

        X['EducationOrdinal'] = X['Education'].map({
          'Basic': 0,
          '2n Cycle': 1,
          'Graduation': 2,
          'Master': 3,
          'PhD': 4
          })

        X.drop(columns=['ID', 'Marital_Status', 'Education'], inplace=True)

        X.drop(columns=['Z_CostContact', 'Z_Revenue'], inplace=True)

        X.drop(columns=['AcceptedCmp1', 'AcceptedCmp2',
                      'AcceptedCmp3', 'AcceptedCmp4',
                      'AcceptedCmp5', 'Complain', 'Response'
                      ], inplace=True)


        X['DaysSinceRegistration'] = (pd.Timestamp.now() -
                                      pd.to_datetime(X['Dt_Customer'],
                                                     format='%d-%m-%Y')
                                     ).dt.days

        X.drop(columns=['Dt_Customer'], inplace=True)

        # remove outliers using IQR
        for col in X.select_dtypes(include='number'):
            X[col] = self.cap_outliers(X[col])

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def cap_outliers(self, series):
        lower, upper = self.iqr_bounds(series)

        return series.clip(lower=lower, upper=upper)

    def iqr_bounds(self, series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        return lower_bound, upper_bound


