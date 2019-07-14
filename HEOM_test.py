import pytest
from HEOM import HEOM
@pytest.fixture
def params():
    return {'data': 0, 'col_ix': 0, 'nan_eqvs': 0, 'result': 0}

def init_test(params):
    metric = HEOM(params['data'], params['col_ix'], params['nan_eqvs'])
    return metric

def heom_test(init_test, params):
    metric = init_test(params)