"""
This file tests the calculate residual helper function, used in .fit().
"""
import numpy as np
import pandas as pd

import sigmet.au3_functions as au3

dates = pd.date_range(start='1/1/2005', periods=12, freq='M')

data_regular = pd.Series(data=[0, -2, -3, -2, -1, 0, 1, 1, 3, -2, -4, -9], index=dates)
reg_regular = pd.Series(data=[0, 1, 1, 2, 1, 0, 1, 2, 3, 4, 5, 7], index=dates)
data_flat = pd.Series(data=np.repeat(100, 12), index=dates)
reg_flat = pd.Series(data=np.repeat(100.1, 12), index=dates)

def test_regular():
    """
    Tests a 'normal' regression, no irregular cases.
    """
    assert 13 == au3.calc_resid(data_regular, reg_regular[1:6],
                                                pd.to_datetime('1/31/2005'),
                                                pd.to_datetime('6/30/2005'))

def test_flat():
    """
    Tests a 'flat' regression, where end date is at the end of the Series.
    """
    assert np.isclose(1.1, au3.calc_resid(data_flat, reg_flat[1:],
                                                pd.to_datetime('1/31/2005'),
                                                pd.to_datetime('12/31/2005')))
