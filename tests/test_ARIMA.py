"""
This file tests the ARIMA helper function, used in .fit().
"""
import numpy as np
import pandas as pd
import scipy.stats
import sigmet.au3_functions as au3
from statsmodels.tsa.arima_model import ARIMA


begin_up = list(range(-12, 0))
begin_down = list(range(12, 0, -1))
begin_flat = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
begins = [begin_up, begin_down, begin_flat]

dates = pd.date_range(start='1/1/2005', periods=24, freq='M')

random = list(np.random.rand(1, 12).flatten())



def prepend_begins(case):
    """
    Adds different data before test case to represent data before start date
    """
    prepended = []

    for begin in begins:
        data = begin.copy()
        data.extend(case)
        ser = pd.Series(data=data, index=dates)
        prepended.append(ser)

    return prepended


def test_index_after_start():
    """
    Tests whether ARIMA forecasts data after start date
    """
    for case in prepend_begins(random):
        assert 12 == len(au3.ARIMA_50(case, dates[12]))


def test_forecast_up():
    """
    Tests whether ARIMA forecast is positive given increasing training data.
    Cases use clean and noisy data to train ARIMA forecast
    """
    basic_data = begin_up.copy()
    basic_data.extend(random)
    basic_up = pd.Series(data=basic_data, index=dates)

    assert all(au3.ARIMA_50(basic_up, dates[12]) >= noisy_up[12])

    noisy_data = [x + y for x, y in zip(begin_up, random)].extend(random)
    noisy_data.extend(random)
    noisy_up = pd.Series(data=noisy_data, index=dates)

    assert all(au3.ARIMA_50(noisy_up, dates[12] >= noisy_up[12]))


def test_forecast_down():
    """
    Tests whether ARIMA forecast is negative given decreasing training data
    Cases use clean and noisy data to train ARIMA forecast
    """
    basic_data = begin_down.copy()
    basic_data.extend(random)
    basic_down = pd.Series(data=basic_data, index=dates)

    assert all(au3.ARIMA_50(basic_up, dates[12]) >= noisy_up[12])

    noisy_data = [x - y for x, y in zip(begin_down, random)]
    noisy_data.extend(random)
    noisy_down = pd.Series(data=noisy_data, index=dates)

    assert all(au3.ARIMA_50(noisy_up, dates[12] <= noisy_up[12]))



def test_forecast_flat():
    """
    Tests whether ARIMA forecast is flat provided flat training data
    """
    data = begin_flat
    data.extend(begin_flat)
    flat_series = pd.Series(data=data, index=dates)
    assert all(au3.ARIMA_50(flat_series, dates[12]) == 0)
