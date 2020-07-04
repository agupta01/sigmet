"""
This file tests the find start date helper function, used in .fit().
"""
import numpy as np
import pandas as pd
import scipy.stats
import sigmet.au3_functions as au3
from statsmodels.tsa.arima_model import ARIMA

dates = pd.date_range(start='1/1/2005', periods=12, freq='M')
increasing = pd.Series(
    data=[1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 12, 15], index=dates)
decreasing = pd.Series(
    data=[3, 2, 1, 0, 0, -1, -2, -5, -6, -8, -9, -15], index=dates)
two_peak_one = pd.Series(data=[0, 4, 0, -4, 0, 6, 0, -6, -5, -4, -2, 0], index=dates)
two_peak_two = pd.Series(
    data=[0, 2, -2, -6, -5, -7, -9, -8, -9, -8, -6, -5], index=dates)
multiple_peaks = pd.Series(
    data=[0, 4, 0, -4, 0, 2, 0, 6, 0, -6, -6, 0], index=dates)


def test_two_peaks():
    """
    Tests two peak cases.
    """
    ARIMA
    assert dates[5] == au3.find_start(two_peak_one, dates[0], dates[-1], 0.001)
    assert dates[1] == au3.find_start(two_peak_two, dates[0], dates[-1], 0.001)


def test_multiple_peaks():
    """
    Tests multiple peaks
    """
    assert dates[1] == au3.find_start(multiple_peaks, dates[0], dates[-1], 0.001)
