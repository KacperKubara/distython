import pytest
import numpy as np 
from HEOM import HEOM

@pytest.fixture
def params():
    return {'data': np.asarray([[1, np.nan, 5], [2, 2, 2], [1, 5, 3]]),
             'col_ix': [0], 'nan_eqvs': [np.nan], 'result': [3.0, 1.44, 2.11]}

def test_init(params):
    metric = HEOM(params['data'], params['col_ix'], params['nan_eqvs'])
    return metric

def test_heom(params):
    metric = test_init(params)
    data = params['data']
    results = params['result']
    assert results[0] == round(metric.heom(data[0], data[1]), 2)
    assert results[1] == round(metric.heom(data[0], data[2]), 2)
    assert results[2] == round(metric.heom(data[1], data[2]), 2)
