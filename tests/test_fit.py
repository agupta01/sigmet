"""
This file tests the .fit() main function of the Sigmet package.
"""
import numpy as np
import pandas as pd

from sigmet.sigmet import Sigmet

dates_dataset1 = pd.date_range(start='1/1/2005', periods=24, freq='M')
dates_dataset2 = pd.date_range(start='1/1/2005', periods=14, freq='M')
dates_dataset3 = pd.date_range(start='1/1/2005', periods=16, freq='M')
dates_dataset4 = pd.date_range(start='1/1/2005', periods=19, freq='M')

dataset1 = pd.Series(data=[
    12, 13, 15, 14, 12, 11.5, 13, 13.5,
    11, 10, 9, 4.5, 2.5, 1, 1, 1.5, 1,
    2, 4, 7, 8, 11, 12.5, 14],
    index=dates_dataset1)

dataset2 = pd.Series(data=[
    4, 0.1, 3, 13.7, 7.5, 4, 0, -6, -2, 
    0.5, 4, 12, 7, 6],
    index=dates_dataset2)

dataset3 = pd.Series(data=[
    99.5, 99, 99.7, 100, 98.5, 96, 92, 90, 
    90.5, 92, 96, 97, 98, 99, 100, 101],
    index=dates_dataset3)

dataset4 = pd.Series(data=[
    145, 146, 148, 149, 150, 152, 154, 151, 
    137, 131, 136, 143, 146, 148, 150, 151, 
    151.5, 152.5, 152],
    index=dates_dataset4)

def test_dataset1():
    """
    Tests dataset1, normal recession
    """
    # Expected: 80.36314052579074
    start_date = dates_dataset1[7] # 13.5
    end_date = dates_dataset1[-1]
    sigmet = Sigmet(dataset1)
    assert np.isclose(80.36314052579074, sigmet.fit(start_date, end_date, force_start=True), atol=1)
    assert np.isclose(80.36314052579074, sigmet.fit(dates_dataset1[2], end_date), atol=1)

def test_dataset2():
    """
    Tests dataset2, normal recession
    """
    start_date = dates_dataset2[1] # 13.7
    end_date = dates_dataset2[-2]
    sigmet = Sigmet(dataset2)
    assert np.isclose(363.29172903368146, sigmet.fit(start_date, end_date, recovery_threshold=0.85), atol=1)

def test_dataset3():
    """
    Tests dataset3, normal recession
    """
    start_date = dates_dataset3[2] # 100
    end_date = dates_dataset3[-1]
    sigmet = Sigmet(dataset3)
    assert np.isclose(45.77548478066123, sigmet.fit(start_date, end_date, recovery_threshold=1), atol=1)

def test_dataset4():
    """
    Tests dataset4, normal recession
    """
    start_date = dates_dataset4[0]
    alt_start = dates_dataset4[5]
    end_date = dates_dataset4[-1]
    sigmet = Sigmet(dataset4)
    assert np.isclose(97.25239080935879, sigmet.fit(alt_start, end_date), atol=1)
