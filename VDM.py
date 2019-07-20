import numpy as np 

class VDM():
    def __init__(self, X, y_ix, cat_ix):
        self.cat_ix = cat_ix
        self.col_ix = [i for i in range(X.shape[1])]
        self.y_ix = y_ix

        self.classes = np.unique(X[:, y_ix])
        print("self.classes: {}".format(self.classes))
        print("X[:, self.cat_ix]: {}".format(X[:, self.cat_ix]))
        
        array_len = 0
        # Get the max no. of unique classes for columns to initialize the array
        for ix in self.cat_ix:
            print("np.unique(X[:, ix]): {}".format(np.unique(X[:, ix])))
            max_val = len(np.unique(X[:, ix]))
            if max_val > array_len:
                array_len = max_val
                print("max_val: {}".format(max_val))
        # Store the list of unique classes elements for each categorical column
        self.unique_attributes = np.full((array_len, len(self.col_ix)), fill_value=-1)
        print(self.unique_attributes)
        for ix in self.cat_ix:
            unique_vals = np.unique(X[:, ix])
            self.unique_attributes[0:len(unique_vals), ix] = unique_vals
            print("unique vals: {}\n self.unique_attributes[:, ix]: {} \n".format(unique_vals, self.unique_attributes[:, ix]))
        
        self.unique_attributes_cnt = np.zeros(len(self.col_ix))
        # Declare the 3D numpy array which holds specifc count for each attribute
        # for each column for each output class
        # +1 in len(self.classes) + 1 is to store the sum (N_a,x) in the last element
        self.final_count = np.zeros((len(self.col_ix), self.unique_attributes.shape[0], len(self.classes) + 1))
        print("Initialized final_count \n {}".format(self.final_count))
        # For each columns
        for i, col in enumerate(self.cat_ix):
            # For each attribute value in the column
            for j, attr in enumerate(self.unique_attributes[col]):
                if attr != -1:
                    # For each output class value
                    for k, val in enumerate(self.classes):
                        # Get an attribute count for each output class
                        row_ix = np.argwhere(X[:, col] == attr)
                        print("np.sum(X[row_ix, y_ix] == val): {}".format(np.sum(X[row_ix, y_ix] == val)))
                        cnt = np.sum(X[row_ix, y_ix] == val)
                        self.final_count[i, j, k] = cnt
                    # Get a sum of all occurences
                    self.final_count[i, j, -1] = np.sum(self.final_count[i, j, :])
        print("cat_ix: {}".format(cat_ix))
        print("unique_attributes: {}".format(self.unique_attributes))
        print("classes: {}".format(self.classes))        
        print("final_count: {}".format(self.final_count))



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
        result = np.zeros(len(x))

        for i in range(len(self.cat_ix)):
            # Get indices to access the final_count array 
            x_ix = np.argwhere(self.final_count[i] == x[i])
            y_ix = np.argwhere(self.final_count[i] == y[i])
            
            N_ax = self.final_count[i, x_ix, :-1]
            N_ay = self.final_count[i, y_ix, :-1]
            N_axc = self.final_count[i, x_ix]
            N_ayc = self.final_count[i, y_ix]

            temp_result = abs(N_ax/N_axc - N_ay/N_ayc)
            temp_result = np.sum(temp_result)

            result[i] = temp_result

        return np.sum(result)
    