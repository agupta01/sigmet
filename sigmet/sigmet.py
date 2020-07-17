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
    
    if standardize == True:
        srs = standardize(srs)

    return self
