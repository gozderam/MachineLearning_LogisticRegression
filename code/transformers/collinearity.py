import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor

class CollinearFeaturesVIFTransformer(BaseEstimator, TransformerMixin):
    '''
    takes datafarme and returns dataframe
    '''
    def __init__(self, vif_threshold=25):
        self.vif_threshold =  vif_threshold

    def fit(self, X, y = None):
        X_vif = X.copy()
        columns_to_remove = []
        while(X_vif.shape[1] > 1):
            vifs = self.__get_vifs(X_vif)
            max_vif = vifs.VIF.max()
            max_vif_feat = vifs.feature[vifs.VIF.idxmax()]
            if max_vif > self.vif_threshold:
                columns_to_remove.append(max_vif_feat)
                X_vif = X_vif.drop([max_vif_feat], axis = 1)
            else:
                break 

        self.features_to_remove_ = columns_to_remove
        return self

    def transform(self, X, y = None):
        return X.drop(self.features_to_remove_, axis = 1)

    def __get_vifs(self, X):
        vifs = pd.DataFrame()
        vifs["feature"] = X.columns
        vifs["VIF"] = [variance_inflation_factor(X.values, i) for i in  range(len(X.columns))]
        return vifs