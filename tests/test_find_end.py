"""
This file tests the find end date helper function, used in .fit().
"""
import numpy as np
import pandas as pd
import sigmet.au3_functions as au3

dates = pd.date_range(start='1/1/2005', periods=12, freq='M')

increasing = pd.Series(data=[1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 12, 15], index=dates)
increasing_forecasted = pd.Series(data=np.repeat(1, 12), index=dates)
decreasing = pd.Series(data=[3, 2, 1, 0, 0, -1, -2, -5, -6, -8, -9, -15], index=dates)
decreasing_forecasted = pd.Series(data=np.repeat(1, 12), index=dates)
one_peak = pd.Series(data=[3, 2, 1, 2, 3, 4, 5, 6, 4, 2, -1, -3], index=dates)
one_peak_forecasted = pd.Series(data=[3, 3, 4, 4, 4, 5, 5, 5, 4, 3, -1, -2], index=dates)
small_peaks = pd.Series(data=[5, 4, 5, 5, 3, 2, 1, 3, 4, 2, 1, 2], index=dates)
small_peaks_forecasted = pd.Series(data=[5, 4, 5, 5, 5, 4, 5, 5, 4, 5, 5, 4], index=dates)

def test_increasing():
    """
    Tests strictly increasing case.
    """
    assert pd.to_datetime('1/31/2005') == au3.find_end(increasing, dates[0], increasing_forecasted)

def test_decreasing():
    """
    Tests strictly decreasing case.
    """
    assert pd.to_datetime('12/31/2005') == au3.find_end(decreasing, dates[2], decreasing_forecasted)

def test_one_peak():
    """
    Tests case with initial dip and full recovery, then larger dip.
    """
    assert dates[6] == au3.find_end(one_peak, dates[0], one_peak_forecasted)

def small_peaks():
    assert dates[11] == au3.find_end(small_peaks, dates[3], small_peaks_forecasted)
