import pytest
import numpy as np 
from VDM import VDM

@pytest.fixture
def params():
    return {'data': np.asarray([[3, 2, 3], [4, 2, 2], [4, 1, 3], [1, 2, 3], [3, 1, 2]]),
            'cat_ix': [1, 2], 'y_ix': [0], 'nan_eqvs': [np.nan],
            'result': [3, 1, 2]}

def test_init(params):
    metric = VDM(params['data'], params['y_ix'], params['cat_ix'])
    return metric

def test_vdm(params):
    metric = test_init(params)
    data = params['data']
    results = params['result']
    final_count = metric.final_count
    #assert results[0] == round(metric.heom(data[0], data[1]), 2)
    #assert results[1] == round(metric.heom(data[0], data[2]), 2)
    #assert results[2] == round(metric.heom(data[1], data[2]), 2)
