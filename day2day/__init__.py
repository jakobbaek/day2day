"""
day2day: Day Trading Application

A comprehensive day trading application for market data collection,
model training, and backtesting trading strategies.
"""

__version__ = "0.1.0"
__author__ = "Jakob Kristensen"
__email__ = "jakobbk@gmail.com"

# Import submodules to ensure they're available
from . import data
from . import models
from . import api
from . import config
from . import backtesting
from . import core