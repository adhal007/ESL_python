from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from typing import List
import numpy as np
from sklearn.preprocessing import StandardScaler
import itertools as itertools
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import pandas as pd


class SplineEstimators(BaseEstimator, TransformerMixin):
    """Generate transformations using different spline estimator.
    Parameters
    ----------
    knots :
        Is a list that contains knots list for each feature.

    target_dofs :
        Is a number of degrees of freedom for each feature. 
        If is set, 
        for natural cubic splines: df+1 knots for features would be automatically calculated at uniform
        quantiles including min and max value of a feature.

        for cubic splines: df knots for features would be automatically calculated at uniform

        for 
    Attributes
    ----------
    dofs_:
        The degrees of freedom for each feature.
    positions_:
        For each feature the index of the first and the last indices of its
        components."""

    def __init__(self, knots: List[List[float]]=None, target_dofs: int = None, method_name: str = 'ns'):

        self.target_dofs = target_dofs
        self.method_name = method_name
        self.available_methods = ['ns', 'cs', 'ls']
        self.knots = knots
        self.quantiles = None

        self._n_knots= None
        if self.method_name not in self.available_methods:
            raise ValueError('Method name is not available')


    @property
    def n_knots(self):
        """
        Property to get the number of knots if not specified using the target degrees of freedom.
        Returns:
        n_knots: int
            The number of knots for each feature.
        """
        if self._n_knots is None:
            if self.method_name == 'ns':
                if self.target_dofs < 1:
                    raise ValueError('The target degrees of freedom should be greater than 1')
                self._n_knots = self.target_dofs + 1
            elif self.method_name == 'cs':
                if self.target_dofs < 4:
                    raise ValueError('The target degrees of freedom should be greater than 2')
                self._n_knots = self.target_dofs - 3
            elif self.method_name == 'ls':
                if self.target_dofs < 2:
                    raise ValueError('The target degrees of freedom should be greater than 1')
                self._n_knots = self.target_dofs - 1
        return self._n_knots
    
    def _calc_dk_ns(self, X:np.ndarray, knot:float, knot_last:float) -> np.ndarray:
        """
        Internal method to calculate the natural cubic spline basis.

        Args:
        X: np.ndarray
            The input data.
        knot: float
            The knot for the feature.
        knot_last: float
            The last knot for the feature.
        Returns:
        np.ndarray
            The natural cubic spline basis.
        """
        return (X - knot).clip(0) ** 3 / (knot_last - knot)

    def _calc_dk_cs(self, X: np.ndarray, knot: float) -> np.ndarray:
        """
        Internal method to calculate the cubic spline basis.

        Args:
        X: np.ndarray
            The input data.
        knot: float

        Returns:
        np.ndarray
            The cubic spline basis.
        """
        zero_arr = np.zeros(X.shape[0]) 
        return np.max(zero_arr, (X - knot)**3)
    
    def _calc_dk_ls(self, X: np.ndarray, knot: float) -> np.ndarray:
        """
        Internal method to calculate the linear spline basis.

        Args:
        X: np.ndarray
            The input data.
        knot: float
            The knot for the feature.
        Returns:
        np.ndarray
            The linear spline basis.
        """
        return np.maximum(0, X - knot)
    
       
    def _calc_expanded_basis_ns(self, X: np.ndarray, knots: np.array) -> np.ndarray:
        """
        Args:
        X: np.ndarray
            The input data ()
        knots: np.array
            The knots for the feature.
        Returns:
        np.ndarray
            The expanded basis for the natural cubic splines.
        """
        X_splines = [X]
        dk_last = self._calc_dk_ns(X, knots[-2], knots[-1])
        ## iterate throguh all knots
        for knot in knots[:-2]:
            dk = self._calc_dk_ns(X, knot, knots[-1])
            X_splines.append(dk - dk_last)
        return np.hstack(X_splines)
    
    def _calc_expanded_basis_cs(self, X: np.ndarray, knots: np.array) -> np.ndarray:
        """
        Internal method to calculate the expanded basis for the cubic splines.

        Args:
        X: np.ndarray
            The input data.
        knots: np.array
            The knots for the feature.
        Returns:
        np.ndarray
            The expanded basis for the cubic splines.
        """ 
        # X_splines = self._add_terms_cs(X, knots)
        X_splines = [X, X**2, X**3]
        for knot in knots:
            X_splines.append(np.maximum(0, (X - knot)**3))
        return np.hstack(X_splines)

    def _calc_expanded_basis_ls(self, X: np.ndarray, knots: np.array) -> np.ndarray:
        """
        Internal method to calculate the expanded basis for the linear splines.

        Args:
        X: np.ndarray
            The input data.
        knots: np.array
            The knots for the feature.
        Returns:
        np.ndarray
            The expanded basis for the linear splines.
        """
        X_splines = [X]
        for knot in knots:
            X_splines.append(self._calc_dk_ls(X, knot))
        return np.hstack(X_splines)
    
    @staticmethod
    def __add_interaction_features(features_basis_splines):
        features_indices = [list(range(expansion.shape[1]))
                            for expansion in features_basis_splines]
        for combination in itertools.product(*features_indices):
            product = features_basis_splines[0][:, combination[0]].copy()
            for i in range(1, len(combination)):
                product *= features_basis_splines[i][:, combination[i]]
            features_basis_splines.append(np.atleast_2d(product).T)

    def _get_quantiles(self, X: np.ndarray) -> np.ndarray:
        """
        Function to get quantiles for the input data.

        Args:
        X: np.ndarray
            The input data.
        Returns:
        quantiles: np.ndarray
            The quantiles for the input data.
        """
        if self.knots is None:
            if self.method_name == 'ns':
                self.quantiles = np.linspace(0.0, 1.0, num=(self.n_knots))
            elif self.method_name == 'cs':
                self.quantiles = np.linspace(0.0, 1.0, num=(self.n_knots+2))[1:-1]
            elif self.method_name == 'ls':
                self.quantiles = np.linspace(0.0, 1.0, num=(self.n_knots+2))[1:-1]
        return self.quantiles
    

    def _compute_knots(self, X: np.ndarray) -> List[List[float]]:
        """
        Calculate knots for natural cubic splines.
        
        Args: 
        X: np.ndarray
            The input data.
        target_dof: int
            The target degrees of freedom.

        Returns:
        List of knots for each feature.
        """
        quantiles = self._get_quantiles(X)    
        knots = []
        for i in range(X.shape[1]):
            knots.append(np.unique(np.quantile(X[:, i], quantiles)))
        return knots
    
    def fit(self, X:np.ndarray, y:np.array=None) -> 'SplineEstimators':
        """
        Method to fit the spline estimator.

        Args:
        X: np.ndarray
            The input data.
        y: np.array
            The target data.
        Returns:
        self: SplineEstimators
        """

        self.knots = self._compute_knots(X)
        if self.method_name == 'ns':
            self.dofs_ = np.array([len(k)-1 for k in self.knots])
        elif self.method_name == 'cs':
            self.dofs_ = np.array([len(k)+3 for k in self.knots])
        elif self.method_name == 'ls':
            self.dofs_ = np.array([len(k) for k in self.knots])

        dofs_cumsum = [0] + list(np.cumsum(self.dofs_))
        self.positions_ = [(dofs_cumsum[i-1], dofs_cumsum[i]-1)
                           for i in range(1, len(dofs_cumsum))]
        return self
                   
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Method to transform the input data using the spline estimator.

        Args:
        X: np.ndarray
            The input data.
        Returns:
        X_spl: np.ndarray
            The transformed data.
        """
        feature_basis_splines = []
        if self.method_name == 'ns':
            
            for i in range(X.shape[1]):
                X_spl = self._calc_expanded_basis_ns(X[:, i:i+1], self.knots[i])
                feature_basis_splines.append(X_spl)
    
        elif self.method_name == 'cs':
            for i in range(X.shape[1]):
                X_spl = self._calc_expanded_basis_cs(X[:, i:i+1], self.knots[i])
                feature_basis_splines.append(X_spl)
        elif self.method_name == 'ls':
            for i in range(X.shape[1]):
                X_spl = self._calc_expanded_basis_ls(X[:, i:i+1], self.knots[i])
                feature_basis_splines.append(X_spl)        
        return np.hstack(feature_basis_splines)
    

    
