import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class OneHotEncodingTransformer(BaseEstimator, TransformerMixin):
    '''
    takes datafarme and returns dataframe
    '''
    def __init__(self, nominal_col_names):
        self.nominal_col_names = nominal_col_names

    def fit(self, X, y = None):
        data_nominal = X.loc[:, self.nominal_col_names]
        self.ohe_ = OneHotEncoder(drop="first")
        self.ohe_.fit(data_nominal)
        return self

    def transform(self, X, y = None):
        data_nominal = X.loc[:, self.nominal_col_names]
        data_one_hot_encoded = self.ohe_.transform(data_nominal).toarray()
        one_hot_encoded_columns_names = self.ohe_.get_feature_names_out(self.nominal_col_names)

        res = pd.concat([
            X.drop(self.nominal_col_names, axis = 1),
            pd.DataFrame(data_one_hot_encoded, columns=one_hot_encoded_columns_names)], axis = 1)

        return res