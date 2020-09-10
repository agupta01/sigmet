"""
This file tests the SARIMAX helper function, used in .fit().
"""
import numpy as np
import pandas as pd
import scipy.stats
import sigmet.au3_functions as au3
from statsmodels.tsa.arima_model import ARIMA


begin_up = list(range(-24, 0))
begin_down = list(range(24, 0, -1))
begins = [begin_up, begin_down]

dates = pd.date_range(start='1/1/2005', periods=48, freq='M')

random = list(np.random.rand(1, 24).flatten())

VALID_THRESHOLD = 22

def prepend_begins(case):
    """
    Adds different data before test case to represent data before start date
    """
    prepended = []

    for begin in begins:
        data = [begin[i] - random[i] + np.sin(i) for i in range(24)]
        data.extend(case)
        ser = pd.Series(data=data, index=dates)
        prepended.append(ser)

    return prepended


def test_index_after_start():
    """
    Tests whether SARIMAX forecasts data after start date
    """

    for case in prepend_begins(random):
        assert 24 == au3.SARIMAX_predictor(case, dates[23], dates[-1]).shape[0]


def test_forecast_up():
    """
    Tests whether SARIMAX forecast is positive given increasing training data.
    Cases use clean and noisy data to train SARIMAX forecast
    """

    noisy_data = [begin_up[i] - random[i] + np.sin(i) for i in range(24)]
    noisy_data.extend(random)
    noisy_up = pd.Series(data=noisy_data, index=dates)

    assert np.count_nonzero(au3.SARIMAX_predictor(noisy_up, dates[24], dates[-1]) > noisy_up[24]) >= VALID_THRESHOLD


def test_forecast_down():
    """
    Tests whether SARIMAX forecast is negative given decreasing training data
    Cases use clean and noisy data to train SARIMAX forecast
    """

    noisy_data = [begin_down[i] - random[i] + np.sin(i) for i in range(24)]
    noisy_data.extend(random)
    noisy_down = pd.Series(data=noisy_data, index=dates)

    assert np.count_nonzero(au3.SARIMAX_predictor(noisy_down, dates[24], dates[-1]) < noisy_down[24]) >= VALID_THRESHOLD
