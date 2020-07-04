import numpy as np
import pandas as pd

dates = pd.date_range(start='1/1/2005', periods=12, freq='M')
increasing = pd.Series(
    data=[1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 12, 15], index=dates)

print(increasing.index.iloc[-1])
