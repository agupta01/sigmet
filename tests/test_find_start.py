"""
This file tests the find start date helper function, used in .fit().
"""
import numpy as np
import pandas as pd
import sigmet.au3_functions as au3


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


def test_flat():
    """
    Tests flat case
    """
    prepended = prepend_begins(flat)
    for case in prepended:
        assert_date(12, case, 12, -1)


def test_increasing():
    """
    Tests strictly increasing case.
    """
    prepended = prepend_begins(increasing)
    for case in prepended:
        assert_date(12, case, 12, -1)


def test_decreasing():
    """
    Tests strictly decreasing case.
    """
    prepended = prepend_begins(decreasing)
    for case in prepended:
        assert_date(12, case, 12, -1)


def test_one_peak():
    """
    Tests one peak cases where peak faces upward and downward respectively
    """
    prepended_upward = prepend_begins(one_peak_upward)
    for case in prepended_upward:
        assert_date(12, case, 12, -1)

    prepended_downward = prepend_begins(one_peak_downward)
    for case in prepended_downward:
        assert_date(12, case, 12, -1)


def test_two_peaks():
    """
    Tests cases where there are two peaks before the recession:
    - Case one: second peak is bigger 
    - Case two: first peak is bigger
    """
    prepended_one = prepend_begins(two_peak_one)
    for case in prepended_one:
        assert_date(17, case, 12, -1)
    
    prepended_two = prepend_begins(two_peak_two)
    for case in prepended_two:
        assert_date(12, case, 12, -1)


def test_multiple_peaks():
    """
    Tests multiple peaks
    """
    assert dates[1] == au3.find_start(multiple_peaks, dates[0], dates[-1], 0.001)
