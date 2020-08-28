from .au3_functions import find_start, SARIMAX_predictor, find_end_baseline, calc_resid
import seaborn as sns
import matplotlib.pyplot as plt

class Sigmet:

    def __init__(self, data):
        # self.start_point = start_point
        # self.end_point = end_point
        self.data = data

    def fit(self, window_start, window_end, sarimax_params=(5, 1, 1), standardize=False, moving_average=1, force_start=False, recovery_threshold=0.9):
        """
        Fits the model and returns a score representing the magnitude of the largest negative shock in the window.

        Parameters
        ----------
        window_start : pd.datetime object
            Date after which to begin searching for a negative shock, exclusive.

        window_end : pd.datetime object
            Date before which to search for a a negative shock, inclusive.

        sarimax_params : tuple (length 3), default=(5, 1, 1)
            (p, q, d) values for tuning SARIMAX model

        standardize : boolean object, default=False
            If False, then no change to the fitted Series.
            If True, then the fitted Series will be standardized before being passed into .fit()
        
        moving_average : int, default=1
            Length of moving average window to apply to data. Default of 1 means
            no MA applied.
            
        force_start : boolean, default=False
            Allows user to 'fix' define start_date of shock instead of searching for it
        
        recovery_threshold : float, default=0.9
            Percentage of starting value (expressed as proportion from 0 to 1) that is considered a "full" recovery

        Returns
        -------
        int
            Returns area score computed from given parameters.
        """

        srs = self.data.copy(deep=True)

        if standardize == True:
            srs = standardize(srs)

        if not force_start:
            start = find_start(srs, window_start, window_end, ma_window=moving_average)
        else:
            start = window_start
            
        end = find_end_baseline(srs, start, window_end, threshold=recovery_threshold)
        sarimax = SARIMAX_predictor(srs, start, end, sarimax_params)
        self.predicted = sarimax
        return calc_resid(srs, sarimax, start, end)

    def graph(self, window_start, window_end, sarimax_params=(5, 1, 1), standardize=False, **kwargs):
        """
        Graphs series along with forecasted trendline starting at recession and ending at end of series.

        Parameters
        ----------
        window_start : pd.datetime object
            Date after which to begin searching for a negative shock.

        window_end : pd.datetime object
            Date before which to search for a a negative shock.

        sarimax_params : tuple (length 3), default=(5, 1, 1)
            (p, q, d) values for tuning SARIMAX model
        
        standardize : boolean object, default=False
            If False, then no change to the fitted Series.
            If True, then the fitted Series will be standardized before being passed into .fit() .
        
        **kwargs: keyword arguments
            args to be passed into the seaborn plot.
        
        Returns
        -------
        Matplotlib.pyplot plot with seaborn styling
        """
        # enables seaborn styling
        sns.set()

        srs = self.data.copy(deep=True)

        if standardize == True:
            srs = standardize(srs)

        start = find_start(srs, window_start, window_end)
        sarimax = SARIMAX_predictor(srs, start, sarimax_params)
        forecasted = srs[srs.index <= start].append(sarimax)
        dy = forecasted - srs
        fig, ax = plt.subplots(2)
        ax[0].plot(srs)
        ax[0].plot(forecasted)
        ax[0].set_xlabel("Time")
                
        ax[1].plot(srs)
        ax[1].scatter(x=srs.index, y=srs)
        ax[1].plot(forecasted)
        ax[1].scatter(x=srs.index, y=forecasted)
        ax[1].fill_between(x=srs.index, y1=srs, y2=forecasted, alpha=0.3, color='gray')
        ax[1].vlines(x=srs.index, ymin=srs, ymax=srs + dy)
        ax[1].set_xlabel("Time")
        return fig
