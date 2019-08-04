import pytest
import numpy as np 
from HVDM import HVDM

@pytest.fixture
def params():
    return {'data': np.asarray([[1, 1, np.nan, 5], [2, 2, 2, 2], [1, 1, 5, 3]]),
             'cat_ix': [1], 'y_ix': [0],'nan_eqvs': [np.nan], 'result': [5.64, 1.16, 4.57]}

def test_init(params):
    metric = HVDM(params['data'], params['y_ix'],params['cat_ix'], params['nan_eqvs'])
    return metric

def test_heom(params):
    metric = test_init(params)
    data = params['data']
    results = params['result']
    assert results[0] == round(metric.hvdm(data[0], data[1]), 2)
    assert results[1] == round(metric.hvdm(data[0], data[2]), 2)
    assert results[2] == round(metric.hvdm(data[1], data[2]), 2)
