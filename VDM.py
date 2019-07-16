import numpy as np 

class VDM():
    def __init__(self, X, y, cat_ix):
        self.cat_ix = cat_ix
        self.col_ix = [i for i in range(X.shape[1])]
        
        self.classes = np.unique(y)
        self.unique_attributes = np.zeros((1, len(self.col_ix)))
        self.unique_attributes_cnt = np.zeros(len(self.col_ix))
        # Get unique classes for each column at once
        self.unique_attributes[:, self.cat_ix] = np.unique(X[self.cat_ix], axis=0)

        # Declare the 3D numpy array holding all of the data which
        # holds specifc count for each attribute for each column for each output class
        # +1 is to store the sum in the last element
        self.final_count = np.zeros((len(self.col_ix), len(max(self.unique_attributes, key=len)[0], len(self.classes) + 1)))
        
        # For each columns
        for i, col in enumerate(self.cat_ix):
            # For each attribute value in the column
            for j, attr in enumerate(self.unique_attributes[col]):
                # For each output class value
                for k, val in enumerate(self.classes):
                    # Get an attribute count for ach output class
                    self.final_count[i, j, k] = X[y == val][:, col]
                    pass
                # Get a sum of all occurences



    def vdm(self, x, y):
        """ Value Difference Metric
        Distance metric function which calculates the distance
        between two instances. Handles heterogeneous data and missing values.
        Uses conditional probability that the output class is given 'c' 
        that attribute 'a' has a value of 'n'.
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
        # count[col][attribute][class] or count[col][attribute][sum] 
        for col in self.col_ix:
            i = np.argwhere(self.unique_attributes[col] == x[col])
            cond_prob_x = self.unique_attributes_cnt[col][i]/self.unique_attributes_cnt[col][-1]
            
            i = np.argwhere(self.unique_attributes[col] == y[col])
            cond_prob_y = self.unique_attributes_cnt[col][i]/self.unique_attributes_cnt[col][-1]

        result = 0
        return result
    