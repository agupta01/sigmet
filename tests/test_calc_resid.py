"""
This file tests the calculate residual helper function, used in .fit().
"""
import sys
import numpy as np

sys.path.insert(1, '../')
import sigmet

data_regular = np.array([0, -2, -3, -2, -1, 0])
reg_regular = np.array([0, 1, 1, 0, 1, 0])
data_flat = np.array([100, 100, 100, 100, 100])
reg_flat = np.array([100.1, 100.1, 100.1, 100.1, 100.1])

def reset_data():
    """
    Resets data to original state, in case it's been modified during a test.
    """
    global data_regular, reg_regular, data_flat, reg_flat
    data_regular = np.array([0, -2, -3, -2, -1, 0])
    reg_regular = np.array([0, 1, 1, 0, 1, 0])
    data_flat = np.array([100, 100, 100, 100, 100])
    reg_flat = np.array([100.1, 100.1, 100.1, 100.1, 100.1])

def test_regular():
    """
    Tests a 'normal' regression, no irregular cases.
    """
    s = Sigmet(data_regular)
    assert 11 == s.calc_resid(data_regular, reg_regular)

def test_flat():
    """
    Tests a 'flat' trendline.
    """
    s = Sigmet(data_flat)
    assert 0.5 == s.calc_resid(data_flat, reg_flat)
