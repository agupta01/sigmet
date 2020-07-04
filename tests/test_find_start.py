"""
This file tests the find start date helper function, used in .fit().
"""
import numpy as np
import pandas as pd
import scipy.stats
import sigmet.au3_functions as au3

initial_up = list(range(-12, 0))
initial_down = list(range(12, 0, -1))
initial_flat = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

dates = pd.date_range(start='1/1/2005', periods=15, freq='M')
flat = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
increasing = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 12, 15]
decreasing = [0, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
two_peak_one = [0, 4, 0, -4, 0, 6, 0, -6, -5, -4, -2, 0]
two_peak_two = [0, 2, -2, -6, -5, -7, -9, -8, -9, -8, -6, -5]
multiple_peaks = [0, 4, 0, -4, 0, 2, 0, 6, 0, -6, -6, 0]

def test_flat():
    """
    Tests flat case
    """
    assert dates[0] == au3.find_start(flat, dates[0], dates[-1], 0.001)

def test_increasing():
    """
    Tests strictly increasing case.
    """
    assert dates[0] == au3.find_start(increasing, dates[0], dates[-1], 0.001)


def test_decreasing():
    """
    Tests strictly decreasing case.
    """
    assert dates[0] == au3.find_start(decreasing, dates[0], dates[-1], 0.001)


def test_normal_curve():
    """
    Tests normal distribution.
    """
    x = np.linspace(0, 12, 12)
    normal_curve_array = scipy.stats.norm.pdf(x, 6, 2)
    normal_curve_series = pd.Series(data=normal_curve_array, index=dates)

    assert dates[0] == au3.find_start(
        normal_curve_series, dates[0], dates[-1], 0.001)

    inverted_curve_array = - 1 * normal_curve_array
    inverted_curve_series = pd.Series(
        data=inverted_curve_array, index=dates)
    assert dates[0] == au3.find_start(
        inverted_curve_series, dates[0], dates[-1], 0.001)


def test_two_peaks():
    """
    Tests two peak cases.
    """
    assert dates[5] == au3.find_start(two_peak_one, dates[0], dates[-1], 0.001)
    assert dates[1] == au3.find_start(two_peak_two, dates[0], dates[-1], 0.001)


def test_multiple_peaks():
    """
    Tests multiple peaks
    """
    assert dates[1] == au3.find_start(multiple_peaks, dates[0], dates[-1], 0.001)
