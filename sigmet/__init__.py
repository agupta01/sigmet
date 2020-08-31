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
import seaborn as sns

plt.style.use("seaborn")

__all__ = ["sigmet", "au3_functions"]