from .au3_functions import find_start, SARIMAX_predictor, find_end_baseline, calc_resid

class Sigmet:

    def __init__(self, start_point, end_point, data):
        self.start_point = start_point
        self.end_point = end_point
        self.data = data

    def fit(self, window_start=None, window_end=None, sarimax_params=(5, 1, 1), standardize=False):
        """
        Fits the model and returns a score representing the magnitude of the largest negative shock in the window.

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

        Returns
        -------
        int
            Returns area score computed from given parameters.
        """
        if window_start == None:
            window_start = self.data.index[0]
        if window_end == None:
            window_end = self.data.index[-1]

        srs = self.data.copy(deep=True)

        if standardize == True:
            srs = standardize(srs)

        start = find_start(srs, window_start, window_end)
        sarimax = SARIMAX_predictor(srs, start, sarimax_params)
        self.predicted = sarimax
        end = find_end_baseline(srs, start, sarimax)
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
        fig, ax = plt.subplots()
        ax.plot(srs)
        ax.plot(forecasted)
        ax.set_xlabel("Time")
        return fig

