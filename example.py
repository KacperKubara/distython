import numpy as np
# Example of how the HEOM metric can be used with Scikit-Learn
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import load_boston
# Importing a custom metric class
from HEOM import HEOM

# Load the dataset from sklearn
boston = load_boston()
boston_data= boston["data"]
# Categorical variables in the data
categorical_ix = [3, 8]

# Introduce some missingness to the data
row_cnt, col_cnt = boston_data.shape
for i in range(row_cnt):
    for j in range(col_cnt):
        rand_val = np.random.randint(20, size=1)
        if rand_val == 10:
            boston_data[i, j] = np.nan

# Declare NearestNeighbor
neighbor = NearestNeighbors()
