import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encodes high-cardinality categorical cols as their frequency (proportion)
    seen in the TRAINING data. Unseen categories at transform time -> 0.

    Must stay identical in behaviour to the class defined in models.ipynb,
    because the pickled pipelines were saved with that class.
    """

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        self.freq_maps_ = {
            col: X[col].value_counts(normalize=True).to_dict()
            for col in self.cols
        }
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.freq_maps_[col]).fillna(0.0)
        return X[self.cols].values

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.cols))