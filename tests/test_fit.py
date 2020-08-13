"""
This file tests the .fit() main function of the Sigmet package.
"""
import numpy as np
import pandas as pd

from sigmet.sigmet import Sigmet

dates = pd.date_range(start='1/1/2005', periods=24, freq='M')

data_regular = pd.Series(data=[
    12, 13, 15, 14, 12, 11.5, 13, 13.5,
    11, 10, 9, 4.5, 2.5, 1, 1, 1.5, 1,
    2, 4, 7, 8, 11, 12.5, 14],
    index=dates)
    
def test_data_regular():
    """
    Tests a 'normal' recession, no irregular cases
    """
    AU3_score = 94.212
    start_date = data_regular[data_regular == 13.5].index
    end_date = data_regular.index[-1]
    sigmet = Sigmet(start_date, end_date, data_regular)
    print(AU3_score, sigmet.fit(start_date, end_date))
    