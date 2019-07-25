# distython
## What is it
It is an implementation of state-of-the-art distance metrics from research papers which can handle mixed-type data and missing values. At the moment, HEOM, HVDM and VDM are tested and working. VDM and HVDM has been released recently so please report bugs, if there are any.
## Why
The aim of the package is to provide ready-to-use heterogeneous distance metrics which are compatible with Scikit-Learn.
Please feel free to help and contribute to the project as there is a lack of existing implementations of hetergeneous distance metrics.
# Installation
**Recommended:** 
`pip install distython`

**Alternatively:**
Clone the repository with `git clone`.
Install the necessary packages with `pipenv install`

# Example - HEOM
```python
# Example code of how the HEOM metric can be used together with Scikit-Learn
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import load_boston
# Importing a custom metric class
from distython import HEOM

# Load the dataset from sklearn
boston = load_boston()
boston_data = boston["data"]
# Categorical variables in the data
categorical_ix = [3, 8]
# The problem here is that NearestNeighbors can't handle np.nan
# So we have to set up the NaN equivalent
nan_eqv = 12345

# Introduce some missingness to the data for the purpose of the example
row_cnt, col_cnt = boston_data.shape
for i in range(row_cnt):
    for j in range(col_cnt):
        rand_val = np.random.randint(20, size=1)
        if rand_val == 10:
            boston_data[i, j] = nan_eqv

# Declare the HEOM with a correct NaN equivalent value
heom_metric = HEOM(boston_data, categorical_ix, nan_equivalents = [nan_eqv])

# Declare NearestNeighbor and link the metric
neighbor = NearestNeighbors(metric = heom_metric.heom)

# Fit the model which uses the custom distance metric 
neighbor.fit(boston_data)

# Return 5-Nearest Neighbors to the 1st instance (row 1)
result = neighbor.kneighbors(boston_data[0].reshape(1, -1), n_neighbors = 5)
print(result)
```
# Research Papers
The code have been implemented based on the following literature:  
-  HEOM, VDM and HVDM: https://arxiv.org/pdf/cs/9701101.pdf
