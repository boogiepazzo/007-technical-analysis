# AGNC Technical Analysis - Forecasting Module
# This module handles advanced forecasting with confidence intervals and Monte Carlo simulation

from config import *

def to_bday_index(series_like, steps, r):
    """Helper: ensure forecast Series has a proper BusinessDay index and correct length"""
    try:
        values = np.asarray(series_like).reshape(-1)
    except Exception:
        values = np.zeros(steps)
    if len(values) != steps:
        # pad or trim to requested steps
        if len(values) < steps:
            pad = np.zeros(steps - len(values))
            values = np.concatenate([values, pad])
        else:
            values = values[:steps]
    idx = pd.bdate_range(start=r.index[-1] + pd.tseries.offsets.BDay(1), periods=steps)
    return pd.Series(values, index=idx)

def get_forecast_with_ci(model, steps, confidence_levels=[0.05, 0.95]):
    """Get forecast with confidence intervals"""
    try:
        forecast = model.get_forecast(steps)
        mean = forecast.predicted_mean
        ci = forecast.conf_int(alpha=0.1)  # 90% confidence interval
        return mean, ci
    except Exception:
        mean = np.zeros(steps)
        ci = pd.DataFrame({'lower': np.zeros(steps), 'upper': np.zeros(steps)})
        return mean, ci

def generate_return_forecasts(ar_res, arma_res, arima_res, r):
    """Generate return forecasts with confidence intervals"""
    print("Generating advanced forecasts with confidence intervals...")

    # Generate forecasts with confidence intervals
    ar_fc_raw, ar_ci = get_forecast_with_ci(ar_res, FORECAST_STEPS)
    arma_fc_raw, arma_ci = get_forecast_with_ci(arma_res, FORECAST_STEPS)
    arima_fc_raw, arima_ci = get_forecast_with_ci(arima_res, FORECAST_STEPS)

    # Convert to proper indexed Series
    ar_fc = to_bday_index(ar_fc_raw, FORECAST_STEPS, r)
    arma_fc = to_bday_index(arma_fc_raw, FORECAST_STEPS, r)
    arima_fc = to_bday_index(arima_fc_raw, FORECAST_STEPS, r)

    # Confidence intervals
    ar_ci_lower = to_bday_index(ar_ci.iloc[:, 0], FORECAST_STEPS, r)
    ar_ci_upper = to_bday_index(ar_ci.iloc[:, 1], FORECAST_STEPS, r)
    arma_ci_lower = to_bday_index(arma_ci.iloc[:, 0], FORECAST_STEPS, r)
    arma_ci_upper = to_bday_index(arma_ci.iloc[:, 1], FORECAST_STEPS, r)
    arima_ci_lower = to_bday_index(arima_ci.iloc[:, 0], FORECAST_STEPS, r)
    arima_ci_upper = to_bday_index(arima_ci.iloc[:, 1], FORECAST_STEPS, r)

    return (ar_fc, arma_fc, arima_fc, 
            ar_ci_lower, ar_ci_upper, 
            arma_ci_lower, arma_ci_upper, 
            arima_ci_lower, arima_ci_upper)

def create_ensemble_forecast(ar_res, arma_res, arima_res, ar_fc, arma_fc, arima_fc):
    """Create ensemble forecast with model weights"""
    print("Creating ensemble forecasts...")

    # Calculate model weights based on AIC (lower AIC = higher weight)
    aic_values = [ar_res.aic, arma_res.aic, arima_res.aic]
    max_aic = max(aic_values)
    weights = [(max_aic - aic + 1) for aic in aic_values]  # Add 1 to avoid zero weights
    weights = np.array(weights) / sum(weights)

    print(f"Model weights: AR={weights[0]:.3f}, ARMA={weights[1]:.3f}, ARIMA={weights[2]:.3f}")

    # Weighted ensemble forecast
    ensemble_fc = weights[0] * ar_fc + weights[1] * arma_fc + weights[2] * arima_fc

    return ensemble_fc, weights

def monte_carlo_price_paths(returns_forecast, volatility_forecast, current_price, n_simulations=1000, steps=FORECAST_STEPS):
    """Generate Monte Carlo price paths"""
    np.random.seed(42)  # For reproducibility
    
    # Generate random shocks
    shocks = np.random.normal(0, 1, (n_simulations, steps))
    
    # Scale by forecasted volatility
    scaled_shocks = shocks * volatility_forecast.values.reshape(1, -1)
    
    # Add forecasted returns
    total_returns = returns_forecast.values.reshape(1, -1) + scaled_shocks
    
    # Convert to price paths
    cumulative_returns = np.cumsum(total_returns, axis=1)
    price_paths = current_price * np.exp(cumulative_returns / 100.0)
    
    return price_paths

def run_monte_carlo_simulation(ar_fc, arma_fc, arima_fc, garch_vol_fc, current_price):
    """Run Monte Carlo simulation for price paths"""
    print("Running Monte Carlo simulations...")

    # Use ensemble forecast for Monte Carlo
    ensemble_returns = (ar_fc + arma_fc + arima_fc) / 3
    ensemble_volatility = garch_vol_fc  # Use GARCH volatility

    # Generate Monte Carlo paths
    mc_paths = monte_carlo_price_paths(ensemble_returns, ensemble_volatility, current_price)

    # Calculate percentiles for fan chart
    percentiles = [5, 10, 25, 75, 90, 95]
    mc_percentiles = np.percentile(mc_paths, percentiles, axis=0)

    return mc_paths, mc_percentiles, percentiles

def calculate_var_es(returns, confidence_levels=[0.05, 0.01]):
    """Calculate Value at Risk and Expected Shortfall"""
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    es_95 = np.mean(returns[returns <= var_95])
    es_99 = np.mean(returns[returns <= var_99])
    
    return {
        'VaR_95': var_95,
        'VaR_99': var_99,
        'ES_95': es_95,
        'ES_99': es_99
    }

def calculate_risk_metrics(r, ensemble_fc):
    """Calculate risk metrics for historical and forecast data"""
    print("Calculating risk metrics...")

    # Calculate risk metrics for historical data
    historical_risk = calculate_var_es(r.values)

    # Calculate risk metrics for forecast
    forecast_risk = calculate_var_es(ensemble_fc.values)

    return historical_risk, forecast_risk

if __name__ == "__main__":
    from data_prep import download_and_prepare_data
    from time_series_models import fit_time_series_models
    from volatility_models import fit_volatility_models, generate_volatility_forecasts
    
    df, r, px, df_raw = download_and_prepare_data()
    ar_res, arma_res, arima_res, arima_order = fit_time_series_models(r, px)
    
    arch_fit, garch_fit, egarch_fit, egarch_available, resid = fit_volatility_models(arma_res, r)
    vol_forecasts = generate_volatility_forecasts(arch_fit, garch_fit, egarch_fit, egarch_available, r)
    
    return_forecasts = generate_return_forecasts(ar_res, arma_res, arima_res, r)
    ensemble_fc, weights = create_ensemble_forecast(ar_res, arma_res, arima_res, 
                                                   return_forecasts[0], return_forecasts[1], return_forecasts[2])
    
    mc_paths, mc_percentiles, percentiles = run_monte_carlo_simulation(
        return_forecasts[0], return_forecasts[1], return_forecasts[2], 
        vol_forecasts[1], float(df['Close'].iloc[-1]))
    
    historical_risk, forecast_risk = calculate_risk_metrics(r, ensemble_fc)
