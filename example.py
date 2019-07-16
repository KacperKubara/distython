import numpy as np
# Example of how the HEOM metric can be used with Scikit-Learn
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import load_boston
# Importing a custom metric class
from HEOM import HEOM

# Load the dataset from sklearn
boston = load_boston()
boston_data = boston["data"]
# Categorical variables in the data
categorical_ix = [3, 8]
# Set up the NaN equivalent
nan_eqv = 12345
# Introduce some missingness to the data
row_cnt, col_cnt = boston_data.shape
for i in range(row_cnt):
    for j in range(col_cnt):
        rand_val = np.random.randint(20, size=1)
        if rand_val == 10:
            boston_data[i, j] = nan_eqv

# Declare the HEOM
heom_metric = HEOM(boston_data, categorical_ix, nan_equvialents = [nan_eqv])

# Declare NearestNeighbor and set metric as heom
neighbor = NearestNeighbors(metric = heom_metric.heom)

# Fit the model
neighbor.fit(boston_data)

# Return 5-Nearest Neighbors to the 1st instance (row 1)
result = neighbor.kneighbors(boston_data[0].reshape(1, -1), n_neighbors = 5)
print(result)