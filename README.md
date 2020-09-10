# sigmet
![Github Tests](https://github.com/agupta01/sigmet/workflows/Python%20Tests/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/sigmet/badge/?version=latest)](https://sigmet.readthedocs.io/en/latest/?badge=latest)
[![GitHub license](https://img.shields.io/github/license/agupta01/sigmet.svg)](https://github.com/agupta01/sigmet/master/LICENSE)

Installation
------------
Sigmet (Signal Metrics Toolkit) is an end-to-end solution for the detection and measurement of negative shocks in time series data.

Sigmet requires the following packages: 

- Pandas: primary data format for easy data manipulation and datetime functionality
- Numpy: for computation
- Statsmodels: AU3 uses statsmodel’s ARIMA and SARIMAX models for prediction
- Matplotlib: plotting for .graph() method



To install, run the following on the command line

```
pip install sigmet
```


Example
-------

First we instantiate an AU3 object

```python
from sigmet import Sigmet
    
data = pd.read_csv('time-series.csv')
ex = Sigmet(start, end, data)
ex.fit(window_start, window_end)

# graph the result as follows
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax = ex.graph()
plt.show()
```

Contribute
----------

- Issue Tracker: https://github.com/agupta01/sigmet/issues
- Source Code: https://github.com/agupta01/sigmet

License
-------

The project is licensed under the GNU General Public License.
