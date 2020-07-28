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


def find_start(series, user_start, user_end, ma_window=6, threshold=1):
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
    threshold : int
        Threshold for local maxima comparison
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
        


def my_min(series, start_date, end_date, threshold=-0.002):
    """Finds an appropriate local minimum to detect a dip accurately, uses
    differencing

    Parameters
    ----------
    series : pd.Series
        Find the correct min within a given time series
    start_date : pd.DateTime
        The start date
    end_date : pd.DateTime
        The end date
    threshold : int
        The differencing threshold for finding the correct min

    Returns
    -------
    str
        The date at which the correct minimum occurs
    """

    # Filter series
    series = series.where(
        series.index > start_date and series.index < end_date)
    my_feature = series.to_numpy()

    # Get min indeces by finding local mins in the numpy array and min vals
    min_indeces = argrelextrema(my_feature, np.less)[0]
    min_vals = [my_feature[val] for val in min_indeces]

    # Get the array of dates at which the minima occur
    min_dates = [series.index.to_numpy()[val] for val in min_indeces]

    # For iteration, update this current index if differencing threshold is
    # reached
    curr_index = 0

    for i in range(len(min_vals)):
        # If we reached the end
        if i == len(min_vals) - 1:
            curr_index = len(min_vals) - 1
            break
        else:
            # Check differencing threshold
            if np.diff([min_vals[i], min_vals[i + 1]]) <= threshold:
                curr_index += 1
            else:
                curr_index = i
                break

    try:
        # Get the dates at this index
        return min_dates[curr_index]
    except ValueError:
        raise "Cannot calculate minimum for given trend"


def ARIMA_50(series, start_date, params=(5, 1, 1)):
    """Get an ARIMA forecast for a given Series

    Parameters
    ----------
    series : pd.Series
        Time-series Series object containing DateTime index
    start_date : pd.DateTime
        DateTime object from index of df representing peak feature
    params : tuple
        p, d, and q parameters for ARIMA

    Returns
    -------
    pd.Series
        Series of forecasts
    """

    try:
        # Filter series
        before = series.where(series.index >= start_date)

        # Steps for ARIMA forecast
        steps = before.values.shape[0]

        # Initialize model
        model = ARIMA(before, order=(5, 1, 1))

        # Fit the model
        model_fit = model.fit(disp=0)

        # Return the forecast as a pd.Series object
        return pd.Series(model_fit.forecast(steps)[0])
    except ValueError:
        raise "Cannot provide an ARIMA forecast for given trend"


def find_end(series, start_date, ARIMA_50):
    """Gets end date of dip in TS, measured as the first point of intersection
    between feature trend and ARIMA_50 foreast for a given element

    Parameters
    ----------
    series : pd.Series
        The input series in which to find the end date
    start_date: pd.datetime
        The start date of the dip
    ARIMA_50 : pd.Series
        ARIMA_50 forecast with which to measure the intersection

    Returns
    -------
    pd.DateTime
        The end date of the dip in TS
    """

    # Calculate differences, use a DataFrame to find the end
    series = series.where(series.index >= start_date)
    diffs = ARIMA_50 - series.values
    residual_df = pd.DataFrame(data={
        'Date': series.index.values, 'Delta': diffs})

    # Filter only positive residuals, and most recent one is the last
    # recession date
    most_recent_positive_delta = residual_df[
        residual_df['Delta'] >= 0].sort_values('Date', ascending=False)

    # If ARIMA model indicates a sharp drop, set end date as one month after
    # start date
    if (most_recent_positive_delta.shape[0] == 0):
        return residual_df.Date.values[0]

    end_date = most_recent_positive_delta['Date'].iloc[0]

    return end_date


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
