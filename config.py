# AGNC Technical Analysis - Configuration and Imports
# This module contains all configuration settings and imports

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys

# Helper function to install packages without pip warnings
def safe_pip_install(package):
    import subprocess
    import sys
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-warn-script-location", package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"Error installing {package}: {e}")

# Ensure required packages are installed
try:
    import yfinance as yf
except ModuleNotFoundError:
    safe_pip_install("yfinance")
    import yfinance as yf

try:
    from statsmodels.tsa.arima.model import ARIMA
except ModuleNotFoundError:
    safe_pip_install("statsmodels")
    from statsmodels.tsa.arima.model import ARIMA

try:
    from arch import arch_model
except ModuleNotFoundError:
    safe_pip_install("arch")
    from arch import arch_model

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ModuleNotFoundError:
    safe_pip_install("matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

# Use reportlab for more reliable PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics import renderPDF
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    import io
    import base64
except ModuleNotFoundError:
    safe_pip_install("reportlab")
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics import renderPDF
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    import io
    import base64

# Set seaborn style
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (12, 8)

# ---------------------------
# Configuration Settings
# ---------------------------
TICKER = "AGNC"
YEARS_BACK = 6          # price history
FIB_LOOKBACK_DAYS = 180 # swing window for Fibonacci levels
FORECAST_STEPS = 30     # AR/ARMA/ARIMA & volatility forecast horizon (business days)

print("Configuration:")
print(f"- Ticker: {TICKER}")
print(f"- Years of data: {YEARS_BACK}")
print(f"- Fibonacci lookback: {FIB_LOOKBACK_DAYS} days")
print(f"- Forecast horizon: {FORECAST_STEPS} business days")

# Global variables to store results
df = None
r = None
px = None
df_raw = None
