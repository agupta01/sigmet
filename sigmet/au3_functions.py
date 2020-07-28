"""Helper Methods for SigMet Time Series Analyzer

This script includes some helper methods to help ensure an accurate and
plausible measure of the shocks within a given time series.

This module accepts a pandas Series and assumes that it contains a DateTime
Index.

This script requires that `pandas`, `numpy`, `tqdm`, `matplotlib`, `sklearn`,
`datetime`, `statsmodels`, and `scipy` be installed within the Python
environment within which you are running this script.

This file can be imported as a module and contains the following functions:

    * standardize
    * find_start
    * my_min
    * ARIMA_50
    * find_end
    * calc_resid
    * find_AU3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima_model import ARIMA
from scipy.signal import argrelextrema
import warnings

plt.style.use("seaborn")


def standardize(series):
    """
    Standardizes the pandas.Series object

    Parameters
    ----------
    series : pandas.Series object
        The pandas.Series object we want to standardize

    Returns
    -------
    standardized_series : pandas.Series object
        Returns a standardized Series
    """

    mu = series.mean()
    sigma = series.std()

    standardized_series = series.subtract(mu).divide(sigma)
    return standardized_series


def find_start(series, user_start, user_end, ma_window=6):
    """Gets start date of dip in TS, measured as the largest local maximum for
    a given element in the database
    Parameters
    ----------
    series : pd.Series
        Time series data, with datetime indices
    user_start : pd.DateTime
        Date after which to start searching for a recession, exclusive
    user_end : pd.DateTime
        Date before which to start searching for a recession, exclusive
    
    Returns
    -------
    pd.DateTime
        The start date of the largest detected recession in window. Returns most recent
        date in window if no recession is found.
    """
    assert(ma_window > 0, "Moving average window cannot be less than 1")
    # filter series to window between start and end date, smooth moving average
    # NOTE: we inclusively filter with user_start b/c when we run the differencing it will no longer be part of the series
    filtered_series = series.rolling(ma_window).mean().loc[(series.index >= user_start) & (series.index < user_end)]
    if filtered_series.hasnans:
        raise ValueError("Moving average value too large for search window.")

    # special case if monotonic decreasing, want to check for largest first derivative
    if filtered_series.is_monotonic_decreasing:
        first_deriv = pd.Series(filtered_series.values[1:-1] - filtered_series.values[2:], index=filtered_series.index[1:-1])
        max_diff = np.argmax(np.abs(first_deriv.values))
        # if multiple diffs with same value, get the one w/ earliest date
        return first_deriv.index[max_diff]
    elif filtered_series.is_monotonic_increasing:
        warnings.warn(UserWarning("Series in window is strictly increasing, no apparent recession. Will return most recent date in window."))
        return filtered_series.index[-1]
    else:
        # find local max and min
        maxes = filtered_series.loc[(filtered_series.shift(1) <= filtered_series) & (filtered_series.shift(-1) < filtered_series)]
        mins = filtered_series.loc[(filtered_series.shift(1) >= filtered_series) & (filtered_series.shift(-1) > filtered_series)]
        # match mins to maxes and sort by amplitudes of recessions
        minmax = pd.DataFrame({})
        if len(mins) == 0:
            mins = filtered_series.loc[(filtered_series.index == filtered_series.index[-1])].copy(deep=True)
        if len(maxes) == 0:
            maxes = filtered_series.loc[(filtered_series.index == filtered_series.index[1])].copy(deep=True)
        if mins.index[0] < maxes.index[0]:
            if len(mins.index[1:]) < len(maxes.index):
                mins = mins.append(pd.Series(filtered_series.loc[filtered_series.index[-1]], index=[filtered_series.index[-1]]))
            minmax = pd.DataFrame({'max': maxes.values, 'min': mins.values[1:]}, index=maxes.index)
        else:
            if len(mins.index) < len(maxes.index):
                mins = mins.append(pd.Series(filtered_series.loc[filtered_series.index[-1]], index=[filtered_series.index[-1]]))
            minmax = pd.DataFrame({'max': maxes.values, 'min': mins.values}, index=maxes.index)
        minmax['height'] = minmax['max'].values - minmax['min'].values
        minmax.sort_values(by='height', ascending=False, inplace=True)
        return minmax.index[0]


def calc_resid(series, predicted, start_date, end_date):
    """Calculates the sum of all residuals between the actual values and
    predicted values between start_date and end_date

    Parameters
    ----------
    series : pd.Series
        Time-series Series object containing DateTime index
    predicted : pd.Series
        Predicted values from max to last date of time-series
    start_date : pd.DateTime
        DateTime object from index of series, representing peak
    end_date : pd.DateTime
        DateTime object from index of series, representing intersection
        of actual and predicted values

    Returns
    -------
    int
        The sum of all residuals between predicted and actual values
    """

    # Filter series
    series = series.loc[(series.index >= start_date) & (series.index <= end_date)]

    # End for filtering predicted
    end_index = len(series)
    predicted_to_end = predicted[:end_index]

    # Get residual lengths via subtraction and add them all up
    diffs = predicted_to_end - series.values
    return sum(diffs)


def find_AU3(series, start_date, end_date, threshold=-0.002):
    """Calculates the AU3 score, given by the area between the ARIMA curve
    and actual curve given by the trend in the series

    Parameters
    ----------
    series : pd.Series
        The series to perform AU3 on
    start_date : pd.DateTime
        The cutoff date
    end_date : pd.DateTime
        The end_date cutoff for find_start (my_min), leverages user to decide
        what to analyze
    threshold : int
        Differencing threshold

    Returns
    -------
    int
        The AU3 score
    """

    start = find_start(series, start_date, end_date, threshold)
    arima = ARIMA_50(series, start)
    end = find_end(series, start, arima)
    return calc_resid(series, arima, start, end)
