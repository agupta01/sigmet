"""Helper Methods for SigMet Time Series Analyzer

This script includes some helper methods to help ensure an accurate and
plausible measure of the shocks within a given time series.

This module accepts a pandas Series and assumes that it contains a DateTime
Index.

This script requires that `pandas`, `numpy`, `tqdm`, `matplotlib`, `sklearn`,
`statsmodels`, and `scipy` be installed within the Python environment within
which you are running this script.

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
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima_model import ARIMA
from scipy.signal import argrelextrema
import warnings
import statsmodels.api as sm
from scipy.signal import argrelextrema, argrelmax

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
    filtered_series = series.rolling(ma_window).mean().loc[(
        series.index >= user_start) & (series.index < user_end)]
    if filtered_series.hasnans:
        raise ValueError("Moving average value too large for search window. Decrease the moving_average value or increase the size of the search window.")

    # special case if monotonic decreasing, want to check for largest first derivative
    if filtered_series.is_monotonic_decreasing:
        first_deriv = pd.Series(
            filtered_series.values[1:-1] - filtered_series.values[2:], index=filtered_series.index[1:-1])
        max_diff = np.argmax(np.abs(first_deriv.values))
        # if multiple diffs with same value, get the one w/ earliest date
        return first_deriv.index[max_diff]
    elif filtered_series.is_monotonic_increasing:
        warnings.warn(UserWarning(
            "Series in window is strictly increasing, no apparent recession. Will return most recent date in window."))
        return filtered_series.index[-1]
    else:
        # find local max and min
        maxes = filtered_series.loc[(filtered_series.shift(1) <= filtered_series) & (
            filtered_series.shift(-1) < filtered_series)]
        mins = filtered_series.loc[(filtered_series.shift(1) >= filtered_series) & (
            filtered_series.shift(-1) > filtered_series)]
        # match mins to maxes and sort by amplitudes of recessions
        minmax = pd.DataFrame({})
        if len(mins) == 0:
            mins = filtered_series.loc[(
                filtered_series.index == filtered_series.index[-1])].copy(deep=True)
        if len(maxes) == 0:
            maxes = filtered_series.loc[(
                filtered_series.index == filtered_series.index[1])].copy(deep=True)
        if mins.index[0] < maxes.index[0]:
            if len(mins.index[1:]) < len(maxes.index):
                mins = mins.append(pd.Series(
                    filtered_series.loc[filtered_series.index[-1]], index=[filtered_series.index[-1]]))
            minmax = pd.DataFrame(
                {'max': maxes.values, 'min': mins.values[1:]}, index=maxes.index)
        else:
            if len(mins.index) < len(maxes.index):
                mins = mins.append(pd.Series(
                    filtered_series.loc[filtered_series.index[-1]], index=[filtered_series.index[-1]]))
            minmax = pd.DataFrame(
                {'max': maxes.values, 'min': mins.values}, index=maxes.index)
        minmax['height'] = minmax['max'].values - minmax['min'].values
        minmax.sort_values(by='height', ascending=False, inplace=True)
        return minmax.index[0]


def ARIMA_predictor(series, start_date, params=(5, 1, 1)):
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
        before = series[series.index <= start_date]

        # Steps for ARIMA forecast
        steps = series.shape[0] - before.values.shape[0]

        # Initialize model
        model = ARIMA(before, order=params)

        # Fit the model
        model_fit = model.fit(disp=0)

        # Return the forecast as a pd.Series object
        return pd.Series(model_fit.forecast(steps)[0])
    except ValueError:
        raise ValueError("Cannot provide an ARIMA forecast for given trend")


def SARIMAX_predictor(series, start_date, end_date, params=(5, 1, 1)):
    """Get an SARIMAX forecast for a given Series

    Parameters
    ----------
    series : pd.Series
        Time-series Series object containing DateTime index
    start_date : pd.DateTime
        DateTime object from index of df representing peak feature
    end_date: pd.DateTime
        end date of detected negative shock, only need to forecast till here
    params : tuple
        p, d, and q parameters for SARIMAX

    Returns
    -------
    pd.Series
        Series of forecasts
    """

    try:
        # Filter series
        before = series[series.index <= start_date]

        # Steps for SARIMAX forecast
        steps = len(series[(series.index >= start_date) & (series.index <= end_date)]) - 1

        # Initialize model
        # model = ARIMA(before, order=params)
        model = sm.tsa.statespace.SARIMAX(before, order=params)

        # Fit the model
        model_fit = model.fit(disp=0)

        # Return the forecast as a pd.Series object
        return pd.Series(model_fit.forecast(steps))
    except ValueError as e:
        raise ValueError("Cannot provide a SARIMAX forecast for given trend: {}".format(e))


def find_end_forecast(series, start_date, user_end, forecasted):
    """
    Gets end date of dip in TS, measured as the first point of intersection
    between feature trend and SARIMAX_50 foreast for a given element

    Parameters
    ----------
    series : pd.Series
        The input series in which to find the end date
    start_date: pd.datetime
        The start date of the dip
    forecasted : pd.Series
        predictor function forecast with which to measure the intersection

    Returns
    -------
    pd.DateTime
        The end date of the dip in TS
    """

    # Calculate differences, use a DataFrame to find the end
    series = series.where(series.index > start_date)
    residuals = series * -1 + forecasted
    # residual_df = pd.DataFrame(data={
    #     'Date': series.index.values, 'Delta': diffs})

    # Filter only positive residuals, and most recent one is the last
    # recession date
    # most_recent_positive_delta = residual_df[
    #     residual_df['Delta'] >= 0].sort_values('Date', ascending=False)

    positive_residuals = residuals[residuals >= 0]
    if positive_residuals.shape[0] == 0:
        return user_end

    return positive_residuals.index[0]


def find_end_baseline(series, start_date, user_end, threshold=0.9):
    """
    Gets end date to stop calculating AU3, measured as the first point in 
    time-series greater than max from find_start. If no point exists, return user end date

    Parameters
    ----------
    series : pd.Series
        The input series in which to find the end date
    start_date: pd.datetime
        Start date identified from find_start
    user_end : pd.datetime
        User specified cutoff to stop calculating AU3 on series
    threshold : float, default=0.9
        Percentage of original value (expressed as proportion from 0 to 1) that is considered a "full" recovery

    Returns
    -------
    pd.DateTime
        The end date of the dip in TS 
    """
    # Index series from identified recession start date to user specified end date
    series_filtered = series[(series.index > start_date) & (series.index < user_end)]

    # Get value at start date
    start_value = series[series.index == start_date].iloc[0]
    cutoff = threshold*start_value

    # Find all values greater than start value after minimum
    recession_min_index = series_filtered[series_filtered == series_filtered.min(
    )].index[0]
    series_after_min = series_filtered[series_filtered.index >=
                                       recession_min_index]
    positive_deltas = series_after_min[series_after_min >= cutoff]

    # If no values greater than start return user end date, else return first value
    if positive_deltas.shape[0] == 0:
        return user_end
    return positive_deltas.index[0]


def calc_resid(series, predicted, recession_start, recession_end):
    """Calculates the sum of all residuals between the actual values and
    predicted values between start_date and end_date

    Parameters
    ----------
    series : pd.Series
        Time-series Series object containing DateTime index
    predicted : pd.Series
        Predicted values from max to last date of time-series
    recession_start : pd.DateTime
        DateTime object from index of series, representing peak
    recession_end : pd.DateTime
        DateTime object from index of series, representing intersection
        of actual and predicted values

    Returns
    -------
    int
        The sum of all residuals between predicted and actual values
    """

    # Filter series
    series = series.loc[(
        series.index > recession_start) & (series.index <= recession_end)]

    # Get residual lengths via subtraction and add them all up
    diffs = predicted - series
    return sum(diffs)
