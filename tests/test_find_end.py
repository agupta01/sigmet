"""
This file tests the find end date helper function, used in .fit().
"""
import numpy as np
import pandas as pd
import sigmet.au3_functions as au3

dates = pd.date_range(start='1/1/2005', periods=12, freq='M')

def test_increasing():
    """
    Tests strictly increasing case.
    """
    increasing = pd.Series(data=[1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 12, 15], index=dates)
    increasing_forecasted = pd.Series(data=np.repeat(1, 12), index=dates)
    assert pd.to_datetime('1/31/2005') == au3.find_end(increasing, dates[0], increasing_forecasted)

def test_decreasing():
    """
    Tests strictly decreasing case.
    """
    decreasing = pd.Series(data=[3, 2, 1, 0, 0, -1, -2, -5, -6, -8, -9, -15], index=dates)
    decreasing_forecasted = pd.Series(data=np.repeat(1, 12), index=dates)
    assert pd.to_datetime('12/31/2005') == au3.find_end(decreasing, dates[2], decreasing_forecasted)

def test_one_peak():
    """
    Tests case with initial dip and full recovery, then larger dip.
    """
    one_peak = pd.Series(data=[3, 2, 1, 2, 3, 4, 5, 6, 4, 2, -1, -3], index=dates)
    one_peak_forecasted = pd.Series(data=[3, 3, 4, 4, 4, 5, 5, 5, 4, 3, -1, -2], index=dates)
    assert dates[6] == au3.find_end(one_peak, dates[0], one_peak_forecasted)

def test_small_peak():
    """
    Tests case with large initial dip then smaller peak without full recovery.
    """
    small_peak = pd.Series(data=[5, 4, 5, 5, 3, 2, 1, 3, 4, 2, 1, 2], index=dates)
    small_peak_forecasted = pd.Series(data=[5, 4, 5, 5, 5, 4, 5, 5, 4, 5, 5, 4], index=dates)
    assert dates[11] == au3.find_end(small_peak, dates[3], small_peak_forecasted)

def test_cubic_recovery():
    """
    Tests case with large dip, then a partial recovery followed by stagnation, then full recovery.
    """
    cubic_recovery = pd.Series(data=[5, 4, 5, 5, 3, 1, 3, 4.9, 4.9, 4.9, 5, 6], index=dates)
    cubic_recovery_forecasted = pd.Series(data=[5, 4, 5, 5, 5, 4, 5, 5, 5, 5, 5, 4], index=dates)
    assert dates[10] == au3.find_end(cubic_recovery, dates[3], cubic_recovery_forecasted)

def test_flat():
    """
    Tests case where trend is flat (no slope).
    """
    flat = pd.Series(data=np.repeat(5, 12), index=dates)
    flat_forecasted = pd.Series(data=np.repeat(5, 12), index=dates)
    assert dates[1] == au3.find_end(flat, dates[0], flat_forecasted)

def test_oscillations():
    """
    Tests an oscillatory trend vs. a 'mirrored' forecast trend.
    """
    oscillation = pd.Series(data=[5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4], index=dates)
    oscillation_forecasted = pd.Series(data=[5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6], index=dates)
    assert dates[1] == au3.find_end(oscillation, dates[0], oscillation_forecasted)
