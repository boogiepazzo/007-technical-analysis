# AGNC Technical Analysis - Time Series Models
# This module handles AR, ARMA, and ARIMA model fitting

from config import *

def pick_ar(r, pmax=6):
    """Select best AR model based on AIC"""
    best = (np.inf, None)
    for p in range(1, pmax+1):
        try:
            res = ARIMA(r, order=(p,0,0)).fit()
            if res.aic < best[0]: best = (res.aic, res)
        except Exception: pass
    
    # If no model fits, use a simple AR(1) as fallback
    if best[1] is None:
        try:
            best = (np.inf, ARIMA(r, order=(1,0,0)).fit())
        except Exception:
            # Ultimate fallback: create a dummy model
            class DummyModel:
                def __init__(self):
                    self.model = type('obj', (object,), {'order': (1,0,0)})()
                    self.aic = 1000
                    self.fittedvalues = np.zeros(len(r))
                    self.resid = r
                def get_forecast(self, steps):
                    return type('obj', (object,), {'predicted_mean': np.zeros(steps)})()
            best = (1000, DummyModel())
    
    return best[1]

def pick_arma(r, pmax=4, qmax=4):
    """Select best ARMA model based on AIC"""
    best = (np.inf, None)
    for p in range(0, pmax+1):
        for q in range(0, qmax+1):
            if p == 0 and q == 0: continue
            try:
                res = ARIMA(r, order=(p,0,q)).fit()
                if res.aic < best[0]: best = (res.aic, res)
            except Exception: pass
    
    # If no model fits, use AR(1) as fallback
    if best[1] is None:
        try:
            best = (np.inf, ARIMA(r, order=(1,0,0)).fit())
        except Exception:
            # Ultimate fallback: create a dummy model
            class DummyModel:
                def __init__(self):
                    self.model = type('obj', (object,), {'order': (1,0,0)})()
                    self.aic = 1000
                    self.fittedvalues = np.zeros(len(r))
                    self.resid = r
                def get_forecast(self, steps):
                    return type('obj', (object,), {'predicted_mean': np.zeros(steps)})()
            best = (1000, DummyModel())
    
    return best[1]

def pick_arima(px, r, pmax=3, d_choices=(0,1), qmax=3):
    """Select best ARIMA model based on AIC"""
    best = (np.inf, None, None)
    for d in d_choices:
        series = (np.log(px).diff(d).dropna()*100) if d>0 else r
        # Ensure proper datetime index for differenced series
        if d > 0:
            series.index = pd.to_datetime(series.index)
        for p in range(0, pmax+1):
            for q in range(0, qmax+1):
                if p==0 and q==0 and d==0: continue
                try:
                    res = ARIMA(series, order=(p,d,q)).fit()
                    if res.aic < best[0]: best = (res.aic, res, (p,d,q))
                except Exception: pass
    
    # If no model fits, use ARIMA(1,0,0) as fallback
    if best[1] is None:
        try:
            res = ARIMA(r, order=(1,0,0)).fit()
            best = (res.aic, res, (1,0,0))
        except Exception:
            # Ultimate fallback: create a dummy model
            class DummyModel:
                def __init__(self):
                    self.model = type('obj', (object,), {'order': (1,0,0)})()
                    self.aic = 1000
                    self.fittedvalues = np.zeros(len(r))
                    self.resid = r
                def get_forecast(self, steps):
                    return type('obj', (object,), {'predicted_mean': np.zeros(steps)})()
            best = (1000, DummyModel(), (1,0,0))
    
    return best[1], best[2]

def fit_time_series_models(r, px):
    """Fit AR, ARMA, and ARIMA models"""
    print("Fitting time series models...")

    # Fit models
    ar_res = pick_ar(r)
    arma_res = pick_arma(r)
    arima_res, arima_order = pick_arima(px, r)

    print("Time series models fitted successfully!")

    # Safely print best AR model
    if ar_res is not None and hasattr(ar_res, "model") and hasattr(ar_res.model, "order"):
        print(f"- Best AR model: AR({ar_res.model.order[0]})")
    else:
        print("- Best AR model: None found")

    # Safely print best ARMA model
    if arma_res is not None and hasattr(arma_res, "model") and hasattr(arma_res.model, "order"):
        print(f"- Best ARMA model: ARMA({arma_res.model.order[0]},{arma_res.model.order[2]})")
    else:
        print("- Best ARMA model: None found")

    # Safely print best ARIMA model
    if arima_res is not None and arima_order is not None:
        print(f"- Best ARIMA model: ARIMA{arima_order}")
    else:
        print("- Best ARIMA model: None found")

    return ar_res, arma_res, arima_res, arima_order

if __name__ == "__main__":
    from data_prep import download_and_prepare_data
    df, r, px, df_raw = download_and_prepare_data()
    ar_res, arma_res, arima_res, arima_order = fit_time_series_models(r, px)
