from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

rooms_idx, bedrooms_idx, population_idx, households_idx = 3, 4, 5, 6
X = pd.read_csv("X.csv")
Y = pd.read_csv("Y.csv")


class CombinedAttributesAdder2(BaseEstimator, TransformerMixin):
    """Treats bedrooms_per_room, rooms_per_household and population_per_houshold as hyperparameters
    and return the array including them as features if set to True accordingly.

    Inherited Classes:
        BaseEstimator (class 'sklearn.base.BaseEstimator'): By inherting this class, we get to two extra methods 
        --get_params() and set_params() that are gonna be useful for hyperparameter tuning.
        
        TransformerMixin (class 'sklearn.base.TransformerMixin'): By inherting this class, we get the 
        fit_transform() method.

    Args:
        add_bedrooms_per_room (bool) = Added as feature to the returned tranformed array when the 
        `fit_transform()` or `transform()` method is being called if set to True.

        add_rooms_per_household (bool) = Added as feature to the returned tranformed array when the 
        `fit_transform()` or `transform()` method is being called if set to True.

        add_population_per_household (bool) = Added as feature to the returned tranformed array when the 
        `fit_transform()` or `transform()` method is being called if set to True. 
    """

    def __init__(self, add_bedrooms_per_room=True, add_rooms_per_household=True,
                 add_population_per_household=True):  # no *args or **kwargs

        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.add_rooms_per_household = add_rooms_per_household
        self.add_population_per_household = add_population_per_household

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_idx] / X[:, rooms_idx]
            X = np.c_[X, bedrooms_per_room]

        if self.add_rooms_per_household:
            rooms_per_household = X[:, rooms_idx] / X[:, households_idx]
            X = np.c_[X, rooms_per_household]

        if self.add_population_per_household:
            population_per_household = X[:, bedrooms_idx] / X[:, rooms_idx]
            X = np.c_[X, population_per_household]

        return X
