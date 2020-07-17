"""
This file tests the end-to-end helper function found in au3_functions.py.
"""
import numpy as np
import pandas as pd
import pytest
import sigmet.au3_functions as au3

dates = pd.date_range(start='1/1/2005', periods=12, freq='M')

def test_increasing():
    """
    Tests case of constantly increasing trend. Should be 0, no shock.
    """
    increasing = pd.Series(data=np.arange(12), index=dates)
    assert 0 == au3.find_AU3(increasing, dates[11], dates[11])

def test_decreasing():
    """
    Tests case of constantly decreasing trend. Should be 0, no shock.
    """
    decreasing = pd.Series(data=np.arange(12, 0, -1), index=dates)
    assert 0 == au3.find_AU3(decreasing, dates[11], dates[11])

def test_flat():
    """
    Tests case of flat (no slope) trend. Should be 0, no shock.
    """
    flat = pd.Series(data=np.repeat(5, 12), index=dates)
    assert 0 == au3.find_AU3(flat, dates[11], dates[11])

def test_low_peaks():
    """
    Tests case of sharp drop, then smaller peaks till period end.
    """
    low_peaks = pd.Series(data=[5, 4, 5, 4, 5, 3, 2, 1, 2, 3, 4, 2], index=dates)
    with pytest.raises(ValueError) as err:
        au3.find_AU3(low_peaks, dates[6], dates[6])
    assert "Cannot provide an ARIMA forecast for given trend" in str(err.value)
