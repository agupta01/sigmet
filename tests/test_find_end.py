"""
This file tests the find end date helper function, used in find_end.
"""
import pandas as pd
import sigmet.au3_functions as au3

dates = pd.date_range(start='1/1/2005', periods=12, freq='M')

pass_baseline = pd.Series(data=[80, 90, 100, 80, 60, 40, 20, 40, 70, 100, 130, 150], index=dates)
no_pass_baseline = pd.Series(
    data=[80, 90, 100, 80, 60, 40, 20, 40, 50, 60, 70, 99], index=dates)


def test_within_range():
    """
    Test date returned is within start_date and user_end
    parameters passed
    """

    calculated_end_date = au3.find_end_baseline(
        pass_baseline, dates[2], dates[11])
    assert dates[2] <= calculated_end_date and dates[11] >= calculated_end_date


def test_after_minimum():
    """
    Test wheter date comes after series minimum
    """

    minimum_date = pass_baseline.idxmin()

    calculated_end_date = au3.find_end_baseline(
        pass_baseline, dates[2], dates[11])

    assert minimum_date <= calculated_end_date


def test_pass_baseline():
    """
    Test value at returned end_date is greater than start_date
    """

    start_value = pass_baseline[2]

    calculated_end_value = pass_baseline[au3.find_end_baseline(
        pass_baseline, dates[2], dates[11])]

    assert start_value <= calculated_end_value


def test_no_baseline_end():
    """
    Test behavior return user_end if series doesn't pass start_date value
    """

    calculated_end_date = au3.find_end_baseline(
        no_pass_baseline, dates[2], dates[11])

    assert dates[11] == calculated_end_date
