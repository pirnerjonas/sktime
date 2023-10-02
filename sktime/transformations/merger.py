# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Implements a merger for panel data.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

__author__ = ["benHeid"]


from sktime.transformations.base import BaseTransformer


class Merger(BaseTransformer):
    """Aggregates Panel data that contain data of overlapping windows of one time series.

    Parameters
    ----------
    method : str
        The method to use for aggregation. Can be one of "mean", "median" or an integer.
    """

    _tags = {
        # scitype:y controls whether internal y can be univariate/multivariate
        # if multivariate is not valid, applies vectorization over variables
        "scitype:y": "Panel",
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
    }

    def __init__(self, method="median"):
        self.method = method

        # leave this as is
        super().__init__()


    def _transform(self, X=None, y=None):
        """Merges the the Panel data by aligning them temporally.


        Parameters
        ----------
        X : pd.DataFrame
            The input panel data.
        y : pd.Series
            ignored

        Returns
        -------
        returns a single time series
        """
        horizon = X.shape[-1]

        if self.method == "mean":
            result = np.nanmean(self._align_temporal(horizon, X), axis=0)
        elif self.method == "median":
            result = np.nanmedian(self._align_temporal(horizon, X), axis=0)
        elif isinstance(self.method, int):
            method = self.method
            assert self.method < horizon and self.method >= -horizon, f"{self.method} must be 'mean','median' or an integer between -horizon and horizon -1."
            result = X[:, method].values
        return result

    def _align_temporal(self, horizon, x):
        r = []
        for i in range(horizon):
            _res = np.concatenate(
                [np.full(fill_value=np.nan, shape=(i,)),
                 x.values[:, i],
                 np.full(fill_value=np.nan, shape=(horizon - 1 - i,))]) 
            r.append(_res)
        return np.stack(r)


    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """

        # todo: set the testing parameters for the estimators

