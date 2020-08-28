"""
This file tests the find start helper function found in au3_functions.py, with real life edge cases data.
"""
import os
import numpy as np
import pandas as pd
import sigmet.au3_functions as au3
import warnings
import pytest


source_directory = os.path.dirname(os.path.abspath(__file__))


def test_case_1():
    """
    Tests find_start's reliability on data set 1.
    """
    test_data_path_1 = os.path.join(source_directory, 'data/testing_csv/edgecase1.csv')
    first_test_csv = pd.read_csv(test_data_path_1)
    first_test_csv.index = pd.to_datetime(first_test_csv['Date'])

    result = au3.find_start(first_test_csv['ZHVI_AllHomes'], pd.to_datetime('1/1/1997'), pd.to_datetime('1/1/2015'), 2)

    assert result == pd.to_datetime('2007-02-28 00:00:00')
    

def test_case_2():
    """
    Tests find_start's reliability on data set 2.
    """
    test_data_path_2 = os.path.join(source_directory, 'data/testing_csv/edgecase2.csv')
    second_test_csv = pd.read_csv(test_data_path_2)
    second_test_csv.index = pd.to_datetime(second_test_csv['Date'])

    result = au3.find_start(second_test_csv['ZHVI_AllHomes'], pd.to_datetime('1/1/2000'), pd.to_datetime('1/1/2012'), 6)

    assert result == pd.to_datetime('2007-11-30 00:00:00')


def test_case_3():
    """
    Tests find_start's reliability on data set 3.
    """
    test_data_path_3 = os.path.join(source_directory, 'data/testing_csv/edgecase3.csv')
    third_test_csv = pd.read_csv(test_data_path_3)
    third_test_csv.index = pd.to_datetime(third_test_csv['Date'])

    result = au3.find_start(third_test_csv['ZHVI_AllHomes'], pd.to_datetime('1/1/2001'), pd.to_datetime('1/1/2015'), 1)

    assert result == pd.to_datetime('2003-08-31 00:00:00')


def test_case_4():
    """
    Tests find_start's reliability on data set 4.
    """
    test_data_path_4 = os.path.join(source_directory, 'data/testing_csv/edgecase4.csv')
    fourth_test_csv = pd.read_csv(test_data_path_4)
    fourth_test_csv.index = pd.to_datetime(fourth_test_csv['Date'])

    result = au3.find_start(fourth_test_csv['ZHVI_AllHomes'], pd.to_datetime('1/1/2000'), pd.to_datetime('1/1/2017'), 6)

    assert result == pd.to_datetime('2007-11-30 00:00:00')


def test_case_5():
    """
    Tests find_start's reliability on data set 5.
    """
    test_data_path_5 = os.path.join(source_directory, 'data/testing_csv/edgecase5.csv')
    fifth_test_csv = pd.read_csv(test_data_path_5)
    fifth_test_csv.index = pd.to_datetime(fifth_test_csv['Date'])

    result = au3.find_start(fifth_test_csv['ZHVI_AllHomes'], pd.to_datetime('7/1/2005'), pd.to_datetime('1/1/2018'), 6)

    assert result == pd.to_datetime('2008-02-29 00:00:00')


def test_case_6():
    """
    Tests find_start's reliability on data set 6.
    """
    test_data_path_6 = os.path.join(source_directory, 'data/testing_csv/edgecase6.csv')
    sixth_test_csv = pd.read_csv(test_data_path_6)
    sixth_test_csv.index = pd.to_datetime(sixth_test_csv['Date'])

    result = au3.find_start(sixth_test_csv['ZHVI_AllHomes'], pd.to_datetime('1/1/2004'), pd.to_datetime('1/1/2018'), 2)

    assert result == pd.to_datetime('2009-07-31 00:00:00')


def test_case_7():
    """
    Tests find_start's reliability on data set 7.
    """
    test_data_path_7 = os.path.join(source_directory, 'data/testing_csv/edgecase7.csv')
    seventh_test_csv = pd.read_csv(test_data_path_7)
    seventh_test_csv.index = pd.to_datetime(seventh_test_csv['Date'])

    result = au3.find_start(seventh_test_csv['ZHVI_AllHomes'], pd.to_datetime('1/1/2008'), pd.to_datetime('1/1/2015'), 1)

    assert result == pd.to_datetime('2009-05-31 00:00:00')