# AGNC Technical Analysis - Plotting Module
# This module handles all plotting functionality

from config import *

def save_plot_to_pdf(fig, title, pdf_content, add_to_pdf_func):
    """Save plot as image and add to PDF content"""
    # Save plot to bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    # Add title and image to PDF content
    add_to_pdf_func(title, "", 'heading')
    
    # Create image from bytes buffer
    img = Image(buf, width=6*inch, height=4*inch)
    pdf_content.append(img)
    pdf_content.append(Spacer(1, 12))
    
    plt.show()  # Show plot individually
    plt.close(fig)  # Close to free memory

def plot_technical_indicators(df, fibs, pdf_content, add_to_pdf_func):
    """Generate individual technical analysis plots"""
    print("Generating individual technical analysis plots...")

    # Plot 1: Price Chart with EMAs and Parabolic SAR
    fig1 = plt.figure(figsize=(14, 8))
    plt.plot(df.index, df["Close"], label="Close Price", linewidth=2, color='#1f77b4')
    plt.plot(df.index, df["EMA12"], label="EMA 12", linewidth=1.5, alpha=0.8, color='#ff7f0e')
    plt.plot(df.index, df["EMA26"], label="EMA 26", linewidth=1.5, alpha=0.8, color='#2ca02c')
    plt.scatter(df.index, df["PSAR"], s=12, label="Parabolic SAR", alpha=0.8, color='red')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_to_pdf(fig1, f'{TICKER} - Price Chart with EMAs and Parabolic SAR', pdf_content, add_to_pdf_func)

    # Plot 2: Ichimoku Cloud
    fig2 = plt.figure(figsize=(14, 8))
    plt.plot(df.index, df["Close"], label="Close Price", linewidth=2, color='#1f77b4')
    plt.plot(df.index, df["Tenkan"], label="Tenkan-sen (9)", linewidth=1.5, color='#ff7f0e')
    plt.plot(df.index, df["Kijun"], label="Kijun-sen (26)", linewidth=1.5, color='#2ca02c')

    # Ichimoku Cloud
    valid_cloud = df[["SpanA","SpanB"]].dropna()
    plt.fill_between(valid_cloud.index, valid_cloud["SpanA"], valid_cloud["SpanB"], 
                    alpha=0.3, label="Ichimoku Cloud", color='lightblue')

    # Naive forward projection for Ichimoku
    try:
        proj_idx = pd.bdate_range(df.index[-1] + pd.tseries.offsets.BDay(1), periods=FORECAST_STEPS)
        tenkan_last = float(df["Tenkan"].dropna().iloc[-1])
        kijun_last = float(df["Kijun"].dropna().iloc[-1])
        spanA_last = float(df[["SpanA"]].dropna().iloc[-1].values)
        spanB_last = float(df[["SpanB"]].dropna().iloc[-1].values)
        plt.plot(proj_idx, np.full(len(proj_idx), tenkan_last), linestyle='--', color='#ff7f0e', alpha=0.6, label='Tenkan proj')
        plt.plot(proj_idx, np.full(len(proj_idx), kijun_last), linestyle='--', color='#2ca02c', alpha=0.6, label='Kijun proj')
        plt.fill_between(proj_idx, np.full(len(proj_idx), spanA_last), np.full(len(proj_idx), spanB_last), alpha=0.15, color='lightblue', label='Cloud proj')
    except Exception:
        pass

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_to_pdf(fig2, f'{TICKER} - Ichimoku Cloud Analysis', pdf_content, add_to_pdf_func)

    # Plot 3: Fibonacci Retracement Levels
    fig3 = plt.figure(figsize=(14, 8))
    plt.plot(df.index, df["Close"], label="Close Price", linewidth=2, color='#1f77b4')

    # Fibonacci levels with different colors
    fib_colors = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'brown']
    swing = df.tail(FIB_LOOKBACK_DAYS)
    for i, (name, level) in enumerate(fibs.items()):
        plt.hlines(level, xmin=swing.index[0], xmax=df.index[-1], 
                  linestyles="--", linewidth=2, color=fib_colors[i], 
                  label=f"Fib {name}")

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_to_pdf(fig3, f'{TICKER} - Fibonacci Retracement Levels', pdf_content, add_to_pdf_func)

    # Plot 4: MACD
    fig4 = plt.figure(figsize=(14, 8))
    plt.plot(df.index, df["MACD"], label="MACD", linewidth=2, color='#1f77b4')
    plt.plot(df.index, df["MACDsig"], label="Signal Line", linewidth=2, color='#ff7f0e')
    plt.bar(df.index, df["MACDhist"], label="Histogram", alpha=0.7, width=1, color='#2ca02c')
    plt.axhline(0, linewidth=1, color='black', alpha=0.5)

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('MACD Value', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_to_pdf(fig4, f'{TICKER} - MACD Analysis', pdf_content, add_to_pdf_func)

    # Plot 5: RSI
    fig5 = plt.figure(figsize=(14, 8))
    plt.plot(df.index, df["RSI14"], label="RSI(14)", linewidth=2, color='#1f77b4')
    plt.axhline(70, linestyle="--", linewidth=2, color='red', alpha=0.8, label='Overbought (70)')
    plt.axhline(30, linestyle="--", linewidth=2, color='green', alpha=0.8, label='Oversold (30)')
    plt.fill_between(df.index, 70, 100, alpha=0.1, color='red', label='Overbought Zone')
    plt.fill_between(df.index, 0, 30, alpha=0.1, color='green', label='Oversold Zone')

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('RSI Value', fontsize=12)
    plt.ylim(0, 100)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_to_pdf(fig5, f'{TICKER} - RSI Analysis', pdf_content, add_to_pdf_func)

    # Plot 6: Stochastic Oscillator
    fig6 = plt.figure(figsize=(14, 8))
    plt.plot(df.index, df["%K"], label="%K (14)", linewidth=2, color='#1f77b4')
    plt.plot(df.index, df["%D"], label="%D (3)", linewidth=2, color='#ff7f0e')
    plt.axhline(80, linestyle="--", linewidth=2, color='red', alpha=0.8, label='Overbought (80)')
    plt.axhline(20, linestyle="--", linewidth=2, color='green', alpha=0.8, label='Oversold (20)')
    plt.fill_between(df.index, 80, 100, alpha=0.1, color='red', label='Overbought Zone')
    plt.fill_between(df.index, 0, 20, alpha=0.1, color='green', label='Oversold Zone')

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stochastic Value', fontsize=12)
    plt.ylim(0, 100)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_to_pdf(fig6, f'{TICKER} - Stochastic Oscillator', pdf_content, add_to_pdf_func)

    print("Basic technical analysis plots completed!")

def plot_forecasts(r, ar_res, arma_res, arima_res, arima_order, return_forecasts, ensemble_fc, pdf_content, add_to_pdf_func):
    """Generate individual forecast plots"""
    print("Generating individual forecast plots...")

    ar_fc, arma_fc, arima_fc, ar_ci_lower, ar_ci_upper, arma_ci_lower, arma_ci_upper, arima_ci_lower, arima_ci_upper = return_forecasts

    # Plot 7: Returns and Time Series Models
    fig7 = plt.figure(figsize=(14, 8))
    plt.plot(r.index, r, label="Actual Returns (%)", alpha=0.7, color='gray', linewidth=1)
    plt.plot(pd.Series(ar_res.fittedvalues, index=r.index), label=f"AR Model Fit", linewidth=2, color='#1f77b4')
    plt.plot(pd.Series(arma_res.fittedvalues, index=r.index), label=f"ARMA Model Fit", linewidth=2, color='#ff7f0e')

    # Handle ARIMA fitted values
    if arima_order[1] == 0:
        arima_fit = pd.Series(arima_res.fittedvalues, index=r.index)
    else:
        dser = (np.log(px).diff(arima_order[1]).dropna()*100)
        arima_fit = pd.Series(arima_res.fittedvalues, index=dser.index)

    plt.plot(arima_fit.index, arima_fit, label=f"ARIMA{arima_order} Model Fit", linewidth=2, color='#2ca02c')

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Log Returns (%)', fontsize=12)
    plt.axhline(0, linewidth=1, color='black', alpha=0.5)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_to_pdf(fig7, f'{TICKER} - Time Series Model Fits', pdf_content, add_to_pdf_func)

    # Plot 8: AR Model Forecast with Confidence Intervals
    fig8 = plt.figure(figsize=(14, 8))
    plt.plot(ar_fc.index, ar_fc, label="AR Forecast", linestyle="-", linewidth=2, color='#1f77b4')
    plt.fill_between(ar_fc.index, ar_ci_lower, ar_ci_upper, alpha=0.3, color='#1f77b4', label="AR 90% CI")

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Returns (%)', fontsize=12)
    plt.axhline(0, linewidth=1, color='black', alpha=0.5)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_to_pdf(fig8, f'{TICKER} - AR Model Forecast', pdf_content, add_to_pdf_func)

    # Plot 9: ARMA Model Forecast with Confidence Intervals
    fig9 = plt.figure(figsize=(14, 8))
    plt.plot(arma_fc.index, arma_fc, label="ARMA Forecast", linestyle="-", linewidth=2, color='#ff7f0e')
    plt.fill_between(arma_fc.index, arma_ci_lower, arma_ci_upper, alpha=0.3, color='#ff7f0e', label="ARMA 90% CI")

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Returns (%)', fontsize=12)
    plt.axhline(0, linewidth=1, color='black', alpha=0.5)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_to_pdf(fig9, f'{TICKER} - ARMA Model Forecast', pdf_content, add_to_pdf_func)

    # Plot 10: ARIMA Model Forecast with Confidence Intervals
    fig10 = plt.figure(figsize=(14, 8))
    plt.plot(arima_fc.index, arima_fc, label="ARIMA Forecast", linestyle="-", linewidth=2, color='#2ca02c')
    plt.fill_between(arima_fc.index, arima_ci_lower, arima_ci_upper, alpha=0.3, color='#2ca02c', label="ARIMA 90% CI")

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Returns (%)', fontsize=12)
    plt.axhline(0, linewidth=1, color='black', alpha=0.5)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_to_pdf(fig10, f'{TICKER} - ARIMA Model Forecast', pdf_content, add_to_pdf_func)

    # Plot 11: Ensemble Forecast
    fig11 = plt.figure(figsize=(14, 8))
    plt.plot(ensemble_fc.index, ensemble_fc, label="Ensemble Forecast", linestyle="-", linewidth=3, color='#d62728')

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Returns (%)', fontsize=12)
    plt.axhline(0, linewidth=1, color='black', alpha=0.5)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_to_pdf(fig11, f'{TICKER} - Ensemble Forecast', pdf_content, add_to_pdf_func)

    print("Individual forecast plots completed!")

def plot_advanced_analysis(df, mc_paths, mc_percentiles, percentiles, garch_fit, garch_vol_fc, 
                          garch_vol_ci_lower, garch_vol_ci_upper, historical_risk, forecast_risk, 
                          weights, ar_res, arma_res, arima_res, ensemble_fc, egarch_available, 
                          egarch_vol_fc, arch_vol_fc, pdf_content, add_to_pdf_func):
    """Generate advanced analysis plots"""
    print("Generating advanced analysis plots...")

    # Plot 12: Monte Carlo Price Fan Chart
    fig12 = plt.figure(figsize=(14, 8))
    forecast_dates = pd.bdate_range(start=df.index[-1] + pd.tseries.offsets.BDay(1), periods=FORECAST_STEPS)
    current_price = float(df['Close'].iloc[-1])

    # Plot historical price
    plt.plot(df.index[-252:], df['Close'].iloc[-252:], label="Historical Price", linewidth=2, color='#1f77b4')

    # Plot fan chart
    colors = ['#ff9999', '#ffcc99', '#ffff99', '#ccffcc', '#99ccff', '#cc99ff']
    for i, (pct, color) in enumerate(zip(percentiles, colors)):
        if i < len(percentiles) - 1:
            plt.fill_between(forecast_dates, mc_percentiles[i], mc_percentiles[i+1], 
                            alpha=0.3, color=color, label=f'{pct}-{percentiles[i+1]}%')

    # Plot median path
    median_path = np.percentile(mc_paths, 50, axis=0)
    plt.plot(forecast_dates, median_path, label="Median Path", linewidth=3, color='red')

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_to_pdf(fig12, f'{TICKER} - Monte Carlo Price Fan Chart', pdf_content, add_to_pdf_func)

    # Plot 13: GARCH Volatility Forecast
    fig13 = plt.figure(figsize=(14, 8))
    garch_sigma = pd.Series(garch_fit.conditional_volatility, index=garch_fit.resid.index)
    plt.plot(garch_sigma.index[-252:], garch_sigma.iloc[-252:], label="GARCH Volatility (In-sample)", linewidth=2, color='#1f77b4')
    plt.plot(garch_vol_fc.index, garch_vol_fc, label="GARCH Forecast", linestyle="-", linewidth=2, color='#ff7f0e')
    plt.fill_between(garch_vol_fc.index, garch_vol_ci_lower, garch_vol_ci_upper, alpha=0.3, color='#ff7f0e', label="GARCH 95% CI")

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Volatility (%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_to_pdf(fig13, f'{TICKER} - GARCH Volatility Forecast', pdf_content, add_to_pdf_func)

    # Plot 14: Risk Metrics Comparison
    fig14 = plt.figure(figsize=(14, 8))
    risk_metrics = ['VaR_95', 'VaR_99', 'ES_95', 'ES_99']
    historical_values = [historical_risk[metric] for metric in risk_metrics]
    forecast_values = [forecast_risk[metric] for metric in risk_metrics]

    x = np.arange(len(risk_metrics))
    width = 0.35

    plt.bar(x - width/2, historical_values, width, label='Historical', color='#1f77b4', alpha=0.8)
    plt.bar(x + width/2, forecast_values, width, label='Forecast', color='#ff7f0e', alpha=0.8)

    plt.ylabel('Returns (%)', fontsize=12)
    plt.xticks(x, risk_metrics, rotation=45)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_to_pdf(fig14, f'{TICKER} - Risk Metrics Comparison', pdf_content, add_to_pdf_func)

    # Plot 15: Model Performance Metrics
    fig15 = plt.figure(figsize=(14, 8))
    models = ['AR', 'ARMA', 'ARIMA']
    aic_values = [ar_res.aic, arma_res.aic, arima_res.aic]

    bars = plt.bar(models, aic_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)

    plt.ylabel('AIC Value', fontsize=12)

    # Add weight annotations
    for i, (bar, weight) in enumerate(zip(bars, weights)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                 f'Weight: {weight:.3f}', ha='center', va='bottom', fontsize=11)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_to_pdf(fig15, f'{TICKER} - Model Performance Metrics', pdf_content, add_to_pdf_func)

    # Plot 16: Scenario Analysis
    fig16 = plt.figure(figsize=(14, 8))
    # Bull, Base, Bear scenarios
    scenarios = ['Bull', 'Base', 'Bear']
    scenario_returns = [
        ensemble_fc + 1.5 * garch_vol_fc,  # Bull: +1.5σ
        ensemble_fc,                        # Base: mean
        ensemble_fc - 1.5 * garch_vol_fc   # Bear: -1.5σ
    ]

    colors = ['#2ca02c', '#1f77b4', '#d62728']
    for scenario, returns, color in zip(scenarios, scenario_returns, colors):
        prices = current_price * np.exp(np.cumsum(returns.values) / 100.0)
        plt.plot(forecast_dates, prices, label=f'{scenario} Scenario', linewidth=2, color=color)

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_to_pdf(fig16, f'{TICKER} - Scenario Analysis', pdf_content, add_to_pdf_func)

    # Plot 17: Volatility Term Structure
    fig17 = plt.figure(figsize=(14, 8))
    plt.plot(garch_vol_fc.index, garch_vol_fc, label="GARCH", linewidth=2, color='#1f77b4')
    if egarch_available:
        plt.plot(egarch_vol_fc.index, egarch_vol_fc, label="EGARCH", linewidth=2, color='#ff7f0e')
    plt.plot(arch_vol_fc.index, arch_vol_fc, label="ARCH", linewidth=2, color='#2ca02c')

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Volatility (%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_to_pdf(fig17, f'{TICKER} - Volatility Term Structure', pdf_content, add_to_pdf_func)

    print("Advanced analysis plots completed!")

if __name__ == "__main__":
    print("Plotting module - run from main.py")
