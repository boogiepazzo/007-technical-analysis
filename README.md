# AGNC Technical Analysis Suite

A comprehensive modular technical analysis and forecasting system for AGNC (American Capital Agency Corp) with advanced statistical modeling, risk assessment, and professional reporting capabilities. The system has been refactored into individual Python modules for better organization, reusability, and maintainability.

## 🚀 Features

### 📊 Technical Analysis
- **Price-based Indicators**: EMAs, MACD, Parabolic SAR, Ichimoku Cloud
- **Momentum Indicators**: RSI, Stochastic Oscillator
- **Support/Resistance**: Fibonacci retracement levels
- **Trend Analysis**: Multiple moving averages and trend identification

### 🔮 Advanced Forecasting
- **Time Series Models**: AR, ARMA, ARIMA with automatic model selection
- **Volatility Modeling**: ARCH, GARCH, EGARCH with confidence intervals
- **Monte Carlo Simulation**: 1,000+ price path scenarios
- **Ensemble Forecasting**: AIC-weighted model combinations
- **Confidence Intervals**: 90% confidence bands for all forecasts

### 📈 Risk Management
- **Value at Risk (VaR)**: 95% and 99% confidence levels
- **Expected Shortfall (ES)**: Tail risk measurement
- **Volatility Forecasting**: Forward-looking volatility term structure
- **Scenario Analysis**: Bull/Base/Bear market scenarios

### 📋 Professional Reporting
- **Individual Plot Display**: All 17 plots shown separately
- **PDF Report Generation**: Comprehensive timestamped reports
- **High-Quality Visualizations**: 300 DPI publication-ready charts
- **Complete Documentation**: All calculations and analysis included

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Required Packages
```bash
pip install numpy pandas matplotlib seaborn yfinance statsmodels arch reportlab
```

## 📖 Usage

### Running the Analysis (Modular System)
1. Run the main script: `python main_working.py`
2. The system will execute all modules in sequence
3. View individual plots as they generate
4. Check the generated PDF report: `AGNC_Technical_Analysis_Report_YYYYMMDD_HHMMSS.pdf`

### Alternative: Jupyter Notebook (Legacy)
1. Open `old/agnc_technical_analysis.ipynb` in Jupyter
2. Run all cells sequentially
3. Note: This is the legacy version - use `main_working.py` for the latest features

### Configuration
Modify these parameters in `config.py`:
- `TICKER`: Stock symbol (default: "AGNC")
- `YEARS_BACK`: Historical data period (default: 6 years)
- `FIB_LOOKBACK_DAYS`: Fibonacci calculation window (default: 180 days)
- `FORECAST_STEPS`: Forecast horizon (default: 30 business days)

## 🏗️ Modular Architecture

The system has been refactored into individual Python modules for better organization and maintainability:

### Core Modules
- **`main_working.py`**: Main execution script that orchestrates all modules
- **`config.py`**: Centralized configuration, imports, and global variables
- **`data_prep.py`**: Data download and initial preparation
- **`technical_indicators.py`**: Technical analysis calculations (RSI, MACD, etc.)
- **`time_series_models.py`**: AR, ARMA, ARIMA model fitting and selection
- **`volatility_models.py`**: ARCH, GARCH, EGARCH volatility modeling
- **`forecasting.py`**: Monte Carlo simulation, ensemble forecasting, and risk metrics
- **`plotting.py`**: All visualization functions and plot generation
- **`pdf_generation.py`**: PDF report generation using ReportLab

### Benefits of Modular Design
- **Reusability**: Individual modules can be imported and used independently
- **Maintainability**: Easier to debug, update, and extend specific functionality
- **Testing**: Each module can be tested independently
- **Scalability**: Easy to add new features or modify existing ones
- **Code Organization**: Clear separation of concerns and responsibilities

## 📊 Output

### Individual Plots (17 total)
1. Price Chart with EMAs and Parabolic SAR
2. Ichimoku Cloud Analysis
3. Fibonacci Retracement Levels
4. MACD Analysis
5. RSI Analysis
6. Stochastic Oscillator
7. Time Series Model Fits
8. AR Model Forecast with Confidence Intervals
9. ARMA Model Forecast with Confidence Intervals
10. ARIMA Model Forecast with Confidence Intervals
11. Ensemble Forecast
12. Monte Carlo Price Fan Chart
13. GARCH Volatility Forecast
14. Risk Metrics Comparison
15. Model Performance Metrics
16. Scenario Analysis
17. Volatility Term Structure

### PDF Report Contents
- **Executive Summary**: Key findings and recommendations
- **Data Overview**: Market statistics and current status
- **Technical Indicators**: Current values and signals
- **Model Performance**: AR/ARMA/ARIMA analysis and weights
- **Forecast Results**: Price targets and confidence intervals
- **Risk Assessment**: VaR, ES, and volatility analysis
- **Trading Recommendations**: AI-driven sentiment analysis
- **All Visualizations**: High-quality plots and charts

## 🔬 Technical Details

### Model Selection
- **Automatic AR/ARMA/ARIMA**: AIC-based model selection
- **Robust Error Handling**: Fallback mechanisms for model fitting
- **Ensemble Weights**: AIC-based weighting for optimal combinations

### Forecasting Methods
- **Confidence Intervals**: Statistical uncertainty quantification
- **Monte Carlo Simulation**: Stochastic price path generation
- **Volatility Modeling**: Multiple GARCH variants for volatility forecasting
- **Scenario Analysis**: Stress testing with different market conditions

### Risk Metrics
- **Historical VaR**: Based on empirical return distribution
- **Forecast VaR**: Forward-looking risk assessment
- **Expected Shortfall**: Conditional tail expectation
- **Volatility Comparison**: Historical vs forecast volatility analysis

## 📈 Key Metrics

### Current Analysis (Example)
- **Current Price**: $9.97
- **30-Day Price Range**: $8.50 - $11.20 (Monte Carlo 90% CI)
- **Most Likely Target**: $9.85 (Median path)
- **Current Volatility**: 1.14% (GARCH)
- **Forecast Volatility**: 1.55% (30-day average)

### Model Performance
- **Best Model**: ARIMA(1,0,0) - AIC: 2,847.32
- **Ensemble Weights**: AR(0.45), ARMA(0.35), ARIMA(0.20)
- **Forecast Accuracy**: Confidence intervals for uncertainty quantification

## 🎯 Trading Signals

### Signal Summary
- **RSI Signal**: NEUTRAL (52.48)
- **MACD Signal**: BULLISH (MACD above Signal)
- **Stochastic Signal**: NEUTRAL
- **Parabolic SAR**: BULLISH (Price above SAR)
- **EMA Trend**: BULLISH (EMA12 above EMA26)
- **Ichimoku Signal**: BULLISH (Price above Cloud)

### Overall Recommendation
- **Sentiment**: BULLISH (67% confidence)
- **Risk Level**: Moderate
- **Action**: Consider long positions with appropriate risk management

## 📁 File Structure

```
007-technical-analysis/
├── main_working.py                   # Main execution script (RECOMMENDED)
├── config.py                        # Configuration and imports
├── data_prep.py                     # Data download and preparation
├── technical_indicators.py           # Technical analysis calculations
├── time_series_models.py             # AR, ARMA, ARIMA model fitting
├── volatility_models.py              # ARCH, GARCH, EGARCH modeling
├── forecasting.py                   # Monte Carlo simulation and risk metrics
├── plotting.py                       # All visualization functions
├── pdf_generation.py                 # PDF report generation
├── old/                             # Legacy files (moved for organization)
│   ├── agnc_technical_analysis.ipynb # Original Jupyter notebook
│   ├── portfolio_data.py            # Legacy data utilities
│   ├── portfolio_main.py            # Legacy main script
│   ├── portfolio_optimization.py    # Legacy optimization
│   ├── portfolio_report.py          # Legacy report generation
│   ├── monte_carlo.py               # Legacy Monte Carlo
│   └── test_imports.py              # Test utilities
├── portfolio_reports_*/              # Generated portfolio reports
├── AGNC_Technical_Analysis_Report_*.pdf # Generated analysis reports
└── README.md                        # This file
```

## 🔄 Updates

### Latest Version Features (v3.0 - Modular Architecture)
- ✅ **Modular Architecture**: Refactored into individual Python modules
- ✅ **Improved Organization**: Legacy files moved to `old/` subfolder
- ✅ **Enhanced PDF Generation**: Fixed ReportLab integration with in-memory image handling
- ✅ **Better Error Handling**: Robust error handling across all modules
- ✅ **Cleaner Codebase**: Separated concerns for better maintainability
- ✅ Individual plot display for better visualization
- ✅ Comprehensive PDF report generation
- ✅ Enhanced forecasting with confidence intervals
- ✅ Monte Carlo simulation with 1,000+ paths
- ✅ Advanced risk metrics (VaR, ES)
- ✅ Ensemble forecasting with AIC weighting
- ✅ Professional-grade visualizations
- ✅ Complete error handling and robustness

## 📞 Support

For questions or issues:
1. Check the console output for error messages
2. Verify all required packages are installed (`pip install numpy pandas matplotlib seaborn yfinance statsmodels arch reportlab`)
3. Ensure stable internet connection for data download
4. Review the generated PDF report for detailed analysis
5. Use `python main_working.py` for the latest modular system

## 📄 License

This project is for educational and research purposes. Please ensure compliance with data usage terms and regulations when using financial data.

---

**Generated**: 2025-09-24  
**Version**: 3.0 - Modular Architecture Suite  
**Status**: Production Ready