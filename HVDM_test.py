import numpy as np 
from VDM import VDM

class HVDM(VDM):
    def __init__(self, X , y_ix, cat_ix):
        # Initialize VDM object
        super.__init__(X, y_ix, cat_ix)

        self.cat_ix = cat_ix
        self.col_ix = [i for i in range(X.shape[1])]
    
    def hvdm(self, x, y):
        """ Heterogeneous Value Difference Metric
        Distance metric function which calculates the distance
        between two instances. Handles heterogeneous data and missing values.
        For categorical variables, it uses conditional probability 
        that the output class is given 'c' when attribute 'a' has a value of 'n'.
        For numerical variables, it uses a normalized Euclidan distance.
        It can be used as a custom defined function for distance metrics
        in Scikit-Learn
        
        Parameters
        ----------
        x : array-like of shape = [n_features]
            First instance 
            
        y : array-like of shape = [n_features]
            Second instance
        Returns
        -------
        result: float
            Returns the result of the distance metrics function
        """
        pass