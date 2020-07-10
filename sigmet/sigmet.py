class Sigmet:

    def __init__(start_point, end_point, data):
        this.start_point = start_point
        this.end_point = end_point
        this.data = data



    def fit(self, cutoff_date, p, q, d, standardize=False):
    """
    Fits the pandas.Series object
    
    Parameters
    __________
    
    cutoff_date : 
    
    p :
    
    q :
    
    d :
    
    standardize : boolean object, default=False
        If False, then no change to the fitted Series.
        If True, then the fitted Series will be standardized before being passed into .fit() .
        
    Returns
    _______
    
    self : object
        Returns an instance of self.
    """

    srs = self.copy(deep=True)
    
    temp = standardize_series(srs, standardize)



    return self
    
    
    
    def standardize_series(series, standardize):
        """
        Standardizes the pandas.Series object
        
        Parameters
        __________
        
        series : pandas.Series object
            A pandas.Series object.
            
        standardize : boolean object, default=False
            If False, then no change to the fitted Series.
            If True, then the fitted Series will be standardized before being passed into .fit() .
            
        Return
        ______
        series : pandas.Series object
            Returns the original fitted Series.
        
        result : pandas.Series object
            Returns a standardized Series.
        """
        
        if standardize == True:
            
            mu = series.mean()
            sigma = series.std()
        
            result = series.subtract(mu).divide(sigma)
            return result
            
        else:
            
            return series