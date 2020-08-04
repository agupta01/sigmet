"""
This file tests the find start date helper function, used in .fit().
"""
import numpy as np
import pandas as pd
import sigmet.au3_functions as au3
<<<<<<< HEAD
import warnings
import pytest


dates = pd.date_range(start='1/1/2005', periods=12, freq='M')

flat = pd.Series(data=np.repeat(5, 12), index=dates)
increasing = pd.Series(data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15], index=dates)
decreasing = pd.Series(data=[0, -1, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10], index=dates)
one_peak_upward = pd.Series(data=[0, 1, 1, 4, 9, 16, 16, 9, 4, 1, 0, 0], index=dates)
one_peak_downward = pd.Series(data=[-x for x in one_peak_upward], index=dates)
two_peak_one = pd.Series(data=[0, 4, 0, -4, 0, 6, 0, -6, -5, -4, -2, 0], index=dates)
two_peak_two = pd.Series(data=[0, 2, -2, -6, -5, -7, -9, -8, -9, -8, -6, -5], index=dates)
multiple_peaks = pd.Series(data=[0, 4, 0, -4, 0, 2, 0, 6, 0, -6, -6, 0], index=dates)
cliff = pd.Series(data=[8, 7, 6, 5, 5, 5, 4, 1, -1, -2, -2, -3], index=dates)


def assert_date(actual_index, series, start_index, end_index, ma_window):
    assert dates[actual_index] == au3.find_start(series, dates[start_index], dates[end_index], ma_window)
=======


begin_up = list(range(-12, 0))
begin_down = list(range(12, 0, -1))
begin_flat = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
begins = [begin_up, begin_flat, begin_down]

dates = pd.date_range(start='1/1/2005', periods=24, freq='M')

flat = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
increasing = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
decreasing = [0, -1, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
one_peak_upward = [0, 1, 1, 4, 9, 16, 16, 9, 4, 1, 0, 0]
one_peak_downward = [-x for x in one_peak_upward]
two_peak_one = [0, 4, 0, -4, 0, 6, 0, -6, -5, -4, -2, 0]
two_peak_two = [0, 2, -2, -6, -5, -7, -9, -8, -9, -8, -6, -5]
multiple_peaks = [0, 4, 0, -4, 0, 2, 0, 6, 0, -6, -6, 0]


def prepend_begins(case):
    prepended = []

    for begin in begins:
        data = begin.copy()
        data.extend(case)
        ser = pd.Series(data=data, index=dates)
        prepended.append(ser)
    
    return prepended


def assert_date(actual_index, series, start_index, end_index):
    MIN_THRESHOLD = 1

    assert dates[actual_index] == au3.find_start(series, dates[start_index], dates[end_index], MIN_THRESHOLD)
>>>>>>> 660f70c98d7f37a02c7275a9ed778206be034b2b


def test_flat():
    """
    Tests flat case
    """
<<<<<<< HEAD
    assert_date(4, flat, 3, -1, 3)
    assert_date(8, flat, 7, -1, 3)
=======
    prepended = prepend_begins(flat)
    for case in prepended:
        assert_date(12, case, 12, -1)
>>>>>>> 660f70c98d7f37a02c7275a9ed778206be034b2b


def test_increasing():
    """
    Tests strictly increasing case.
    """
<<<<<<< HEAD
    assert_date(-2, increasing, 1, -1, 1)
    with pytest.warns(UserWarning):
        assert_date(-4, increasing, 2, -3, 2)
=======
    prepended = prepend_begins(increasing)
    for case in prepended:
        assert_date(12, case, 12, -1)
>>>>>>> 660f70c98d7f37a02c7275a9ed778206be034b2b


def test_decreasing():
    """
    Tests strictly decreasing case.
    """
<<<<<<< HEAD
    assert_date(2, decreasing, 0, -1, 1)
    assert_date(4, decreasing, 2, 11, 3)
=======
    prepended = prepend_begins(decreasing)
    for case in prepended:
        assert_date(12, case, 12, -1)
>>>>>>> 660f70c98d7f37a02c7275a9ed778206be034b2b


def test_one_peak():
    """
<<<<<<< HEAD
    Tests one peak upward and downward-facing cases.
    """
    assert_date(7, one_peak_upward, 4, -1, 4)
    assert_date(2, one_peak_downward, 0, -1, 1)
=======
    Tests one peak cases where peak faces upward and downward respectively
    """
    prepended_upward = prepend_begins(one_peak_upward)
    for case in prepended_upward:
        assert_date(12, case, 12, -1)

    prepended_downward = prepend_begins(one_peak_downward)
    for case in prepended_downward:
        assert_date(12, case, 12, -1)
>>>>>>> 660f70c98d7f37a02c7275a9ed778206be034b2b


def test_two_peaks():
    """
    Tests cases where there are two peaks before the recession:
    - Case one: second peak is bigger 
    - Case two: first peak is bigger
    """
<<<<<<< HEAD
    assert_date(5, two_peak_one, 0, -1, 1)
    assert_date(5, two_peak_one, 4, 7, 1)
    assert_date(6, two_peak_one, 1, -1, 2)

    assert_date(1, two_peak_two, 0, -1, 1)
    assert_date(4, two_peak_two, 3, 9, 1)
    assert_date(3, two_peak_two, 2, -1, 3)
=======
    prepended_one = prepend_begins(two_peak_one)
    for case in prepended_one:
        assert_date(17, case, 12, -1)
    
    prepended_two = prepend_begins(two_peak_two)
    for case in prepended_two:
        assert_date(12, case, 12, -1)
>>>>>>> 660f70c98d7f37a02c7275a9ed778206be034b2b


def test_multiple_peaks():
    """
<<<<<<< HEAD
    Tests multiple peaks.
    """
    assert_date(7, multiple_peaks, 0, -1, 1)

def test_cliff():
    """
    Tests monotonous decreasing with a 'cliff'.
    """
    assert_date(6, cliff, 0, -1, 1)
=======
    Tests multiple peaks
    """
    assert dates[1] == au3.find_start(multiple_peaks, dates[0], dates[-1], 0.001)
>>>>>>> 660f70c98d7f37a02c7275a9ed778206be034b2b
