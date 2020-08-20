Welcome to Sigmet!
==================================

.. toctree::
   :maxdepth: 2
   
   api_reference

* :ref:`genindex`
* :ref:`search`

Installation
------------

.. TODO add entire section as code example in start page?
.. add separate page as start page and index pages?

Sigmet requires the following packages: 

- Pandas: primary data format for easy data manipulation and datetime functionality
- Numpy: for computation
- Statsmodels: AU3 uses statsmodel’s ARIMA and SARIMAX models for prediction
- Matplotlib: plotting for .graph() method

.. TODO add names as links


To install, run the following on the command line

.. code-block:: console

    pip install sigmet


Example
-------

First we instantiate an AU3 object

.. code-block:: python
    
   from sigmet import Sigmet
    
   data = pd.read_csv('time-series.csv')
   ex = Sigmet(start, end, data)
   ex.fit(window_start, window_end)


Contribute
----------

- Issue Tracker: https://github.com/agupta01/sigmet/issues
- Source Code: https://github.com/agupta01/sigmet

License
-------

The project is licensed under the GNU General Public License.

