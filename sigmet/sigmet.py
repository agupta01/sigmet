import au3_functions

class Sigmet:

    def __init__(start_point, end_point, data):
        this.start_point = start_point
        this.end_point = end_point
        this.data = data



    def fit(self, start, end, sarimax_params=(5, 1, 1), standardize=False):
    """
    Fits the model and returns a score representing the magnitude of the largest negative shock in the window.
    
    Parameters
    __________
    
    start : pd.datetime object
        Date after which to begin searching for a negative shock.
    
    end : pd.datetime object
        Date before which to search for a a negative shock.
    
    sarimax_params : tuple (length 3), default=(5, 1, 1)
        (p, q, d) values for tuning SARIMAX model
    
    standardize : boolean object, default=False
        If False, then no change to the fitted Series.
        If True, then the fitted Series will be standardized before being passed into .fit() .
        
    Returns
    _______

    int
        Returns area score computed from given parameters.
    """

    srs = self.data.copy(deep=True)
    
    if standardize == True:
        srs = standardize(srs)

    start = au3_functions.find_start(series, window_start, window_end, threshold)
    arima = au3_functions.SARIMAX_50(series, start)
    end = au3_functions.find_end_baseline(series, start, arima)
    return au3_functions.calc_resid(series, arima, start, end)
