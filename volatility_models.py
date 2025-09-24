# AGNC Technical Analysis - Volatility Models
# This module handles ARCH/GARCH volatility modeling

from config import *

def fit_volatility_models(arma_res, r):
    """Fit ARCH/GARCH volatility models"""
    print("Fitting enhanced volatility models...")

    # Use ARMA residuals for volatility modeling
    try:
        # Handle different types of residuals (Series, array, etc.)
        if hasattr(arma_res.resid, 'values'):
            resid_values = arma_res.resid.values
        else:
            resid_values = arma_res.resid
        
        # Ensure we have a 1D array
        if resid_values.ndim > 1:
            resid_values = resid_values.flatten()
        
        resid = pd.Series(resid_values, index=r.index).dropna()
    except Exception as e:
        print(f"Warning: Could not extract residuals properly: {e}")
        # Fallback: use returns directly
        resid = r.dropna()

    # Fit multiple volatility models
    arch_fit  = arch_model(resid, vol="ARCH",  p=1, dist="normal").fit(disp="off")
    garch_fit = arch_model(resid, vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")

    # Try EGARCH for asymmetric volatility
    try:
        egarch_fit = arch_model(resid, vol="EGARCH", p=1, o=1, q=1, dist="normal").fit(disp="off")
        egarch_available = True
    except Exception:
        egarch_fit = garch_fit  # Fallback to GARCH
        egarch_available = False

    return arch_fit, garch_fit, egarch_fit, egarch_available, resid

def get_volatility_forecast_with_ci(model, steps):
    """Get volatility forecast with confidence intervals"""
    try:
        forecast = model.forecast(horizon=steps, reindex=False)
        variance = forecast.variance.values[-1]
        vol_mean = np.sqrt(variance)
        
        # Approximate confidence intervals (simplified)
        vol_std = vol_mean * 0.1  # Assume 10% uncertainty
        vol_lower = vol_mean - 1.96 * vol_std
        vol_upper = vol_mean + 1.96 * vol_std
        
        return vol_mean, vol_lower, vol_upper
    except Exception:
        vol_mean = np.zeros(steps)
        vol_lower = np.zeros(steps)
        vol_upper = np.zeros(steps)
        return vol_mean, vol_lower, vol_upper

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

def generate_volatility_forecasts(arch_fit, garch_fit, egarch_fit, egarch_available, r):
    """Generate volatility forecasts"""
    # Get volatility forecasts
    arch_vol_arr, arch_vol_lower, arch_vol_upper = get_volatility_forecast_with_ci(arch_fit, FORECAST_STEPS)
    garch_vol_arr, garch_vol_lower, garch_vol_upper = get_volatility_forecast_with_ci(garch_fit, FORECAST_STEPS)

    if egarch_available:
        egarch_vol_arr, egarch_vol_lower, egarch_vol_upper = get_volatility_forecast_with_ci(egarch_fit, FORECAST_STEPS)
    else:
        egarch_vol_arr = garch_vol_arr
        egarch_vol_lower = garch_vol_lower
        egarch_vol_upper = garch_vol_upper

    # Convert to Series
    arch_vol_fc = to_bday_index(arch_vol_arr, FORECAST_STEPS, r)
    garch_vol_fc = to_bday_index(garch_vol_arr, FORECAST_STEPS, r)
    egarch_vol_fc = to_bday_index(egarch_vol_arr, FORECAST_STEPS, r)

    arch_vol_ci_lower = to_bday_index(arch_vol_lower, FORECAST_STEPS, r)
    arch_vol_ci_upper = to_bday_index(arch_vol_upper, FORECAST_STEPS, r)
    garch_vol_ci_lower = to_bday_index(garch_vol_lower, FORECAST_STEPS, r)
    garch_vol_ci_upper = to_bday_index(garch_vol_upper, FORECAST_STEPS, r)

    return (arch_vol_fc, garch_vol_fc, egarch_vol_fc, 
            arch_vol_ci_lower, arch_vol_ci_upper, 
            garch_vol_ci_lower, garch_vol_ci_upper)

if __name__ == "__main__":
    from data_prep import download_and_prepare_data
    from time_series_models import fit_time_series_models
    
    df, r, px, df_raw = download_and_prepare_data()
    ar_res, arma_res, arima_res, arima_order = fit_time_series_models(r, px)
    
    arch_fit, garch_fit, egarch_fit, egarch_available, resid = fit_volatility_models(arma_res, r)
    vol_forecasts = generate_volatility_forecasts(arch_fit, garch_fit, egarch_fit, egarch_available, r)
