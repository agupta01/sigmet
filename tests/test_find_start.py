"""
This file tests the find start date helper function, used in .fit().
"""
import numpy as np
import pandas as pd
import sigmet.au3_functions as au3
import warnings
import pytest


dates = pd.date_range(start='1/1/2005', periods=12, freq='M')

flat = pd.Series(data=np.repeat(5, 12), index=dates)
increasing = pd.Series(
    data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15], index=dates)
decreasing = pd.Series(
    data=[0, -1, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10], index=dates)
one_peak_upward = pd.Series(
    data=[0, 1, 1, 4, 9, 16, 16, 9, 4, 1, 0, 0], index=dates)
one_peak_downward = pd.Series(data=[-x for x in one_peak_upward], index=dates)
two_peak_one = pd.Series(
    data=[0, 4, 0, -4, 0, 6, 0, -6, -5, -4, -2, 0], index=dates)
two_peak_two = pd.Series(
    data=[0, 2, -2, -6, -5, -7, -9, -8, -9, -8, -6, -5], index=dates)
multiple_peaks = pd.Series(
    data=[0, 4, 0, -4, 0, 2, 0, 6, 0, -6, -6, 0], index=dates)
cliff = pd.Series(data=[8, 7, 6, 5, 5, 5, 4, 1, -1, -2, -2, -3], index=dates)


def assert_date(actual_index, series, start_index, end_index, ma_window):
    assert dates[actual_index] == au3.find_start(
        series, dates[start_index], dates[end_index], ma_window)


def test_flat():
    """
    Tests flat case
    """
    assert_date(4, flat, 3, -1, 3)
    assert_date(8, flat, 7, -1, 3)


def test_increasing():
    """
    Tests strictly increasing case.
    """
    assert_date(-2, increasing, 1, -1, 1)
    with pytest.warns(UserWarning):
        assert_date(-4, increasing, 2, -3, 2)


def test_decreasing():
    """
    Tests strictly decreasing case.
    """
    assert_date(2, decreasing, 0, -1, 1)
    assert_date(4, decreasing, 2, 11, 3)


def test_one_peak():
    """
    Tests one peak upward and downward-facing cases.
    """
    assert_date(7, one_peak_upward, 4, -1, 4)
    assert_date(2, one_peak_downward, 0, -1, 1)


def test_two_peaks():
    """
    Tests cases where there are two peaks before the recession:
    - Case one: second peak is bigger 
    - Case two: first peak is bigger
    """
    assert_date(5, two_peak_one, 0, -1, 1)
    assert_date(5, two_peak_one, 4, 7, 1)
    assert_date(6, two_peak_one, 1, -1, 2)

    assert_date(1, two_peak_two, 0, -1, 1)
    assert_date(4, two_peak_two, 3, 9, 1)
    assert_date(3, two_peak_two, 2, -1, 3)


def test_multiple_peaks():
    """
    Tests multiple peaks.
    """
    assert_date(7, multiple_peaks, 0, -1, 1)


def test_cliff():
    """
    Tests monotonous decreasing with a 'cliff'.
    """
    assert_date(6, cliff, 0, -1, 1)
