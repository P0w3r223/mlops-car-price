"""mlops-car-price: keeping the A3 used-car price model alive, measured and replaceable.

The modelling code is not reimplemented here — it is consumed from the ``car_price_ml``
package (project A3, pinned to a tag). This package adds the layer around it: versioned
data, recorded training runs, drift monitoring, and a promotion rule that decides when a
challenger may replace the champion.
"""

__version__ = "0.1.0"
