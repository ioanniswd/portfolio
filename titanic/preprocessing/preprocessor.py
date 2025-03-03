import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class Preprocessor:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X.drop(['Cabin'], axis=1, inplace=True)
        X['Age'] = X['Age'].fillna(X['Age'].median())

        titles = X['Name'].str.extract(r',\s(.*?)\.')[0]
        titles.value_counts(dropna=False)

        X['Title'] = np.where(
                titles.isin(['Mr', 'Miss', 'Mrs', 'Master']),
                titles,
                'Other'
                )

        X.drop(columns=['Name'], inplace=True)
        X.drop(columns=['Ticket'], inplace=True)

        X = pd.get_dummies(
                X,
                columns=['Sex', 'Embarked', 'Title'],
                drop_first=False
                ).astype(float)

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
