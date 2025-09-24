# AGNC Technical Analysis - Technical Indicators
# This module contains all technical indicator functions and calculations

from config import *

def ema(series, n):
    """Exponential Moving Average"""
    return series.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    """Relative Strength Index"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def stochastic_k(high, low, close, n=14):
    """Stochastic %K"""
    low_n = low.rolling(n).min()
    high_n = high.rolling(n).max()
    return 100 * (close - low_n) / (high_n - low_n)

def parabolic_sar(high, low, step=0.02, max_step=0.20):
    """Parabolic Stop and Reverse"""
    sar = np.zeros(len(high))
    bull = True
    af = step
    ep = float(low.iloc[0])
    sar[0] = float(low.iloc[0]) - (float(high.iloc[0]) - float(low.iloc[0]))
    for i in range(1, len(high)):
        prev = sar[i-1]
        if bull:
            sar[i] = prev + af*(ep - prev)
            sar[i] = min(sar[i], float(low.iloc[i-1]), float(low.iloc[i]))
            if float(high.iloc[i]) > ep:
                ep = float(high.iloc[i]); af = min(af + step, max_step)
            if float(low.iloc[i]) < sar[i]:
                bull = False; sar[i] = ep; ep = float(low.iloc[i]); af = step
        else:
            sar[i] = prev + af*(ep - prev)
            sar[i] = max(sar[i], float(high.iloc[i-1]), float(high.iloc[i]))
            if float(low.iloc[i]) < ep:
                ep = float(low.iloc[i]); af = min(af + step, max_step)
            if float(high.iloc[i]) > sar[i]:
                bull = True; sar[i] = ep; ep = float(high.iloc[i]); af = step
    return pd.Series(sar, index=high.index)

def ichimoku(df, conv=9, base=26, span_b=52, disp=26):
    """Ichimoku Cloud indicators"""
    H, L, C = df["High"], df["Low"], df["Close"]
    tenkan = (H.rolling(conv).max() + L.rolling(conv).min())/2
    kijun  = (H.rolling(base).max() + L.rolling(base).min())/2
    spanA  = ((tenkan + kijun)/2).shift(disp)
    spanB  = ((H.rolling(span_b).max() + L.rolling(span_b).min())/2).shift(disp)
    chikou = C.shift(-disp)
    return tenkan, kijun, spanA, spanB, chikou

def compute_technical_indicators(df):
    """Compute all technical indicators"""
    print("Computing technical indicators...")

    # MACD (Moving Average Convergence Divergence)
    df["EMA12"] = ema(df["Close"], 12)
    df["EMA26"] = ema(df["Close"], 26)
    df["MACD"]  = df["EMA12"] - df["EMA26"]
    df["MACDsig"] = ema(df["MACD"], 9)
    df["MACDhist"] = df["MACD"] - df["MACDsig"]

    # RSI (Relative Strength Index)
    df["RSI14"] = rsi(df["Close"], 14)

    # Stochastic Oscillator
    df["%K"] = stochastic_k(df["High"], df["Low"], df["Close"], 14)
    df["%D"] = df["%K"].rolling(3).mean()

    # Parabolic SAR
    df["PSAR"] = parabolic_sar(df["High"], df["Low"])

    # Ichimoku Cloud
    df["Tenkan"], df["Kijun"], df["SpanA"], df["SpanB"], df["Chikou"] = ichimoku(df)

    # Fibonacci Retracement Levels
    swing = df.tail(FIB_LOOKBACK_DAYS)
    ph, pl = swing["High"].max(), swing["Low"].min()
    extent = ph - pl
    fibs = {
        "0.0%": ph,
        "23.6%": ph - 0.236*extent,
        "38.2%": ph - 0.382*extent,
        "50.0%": ph - 0.500*extent,
        "61.8%": ph - 0.618*extent,
        "78.6%": ph - 0.786*extent,
        "100%": pl
    }

    print("Technical indicators computed successfully!")
    print(f"\nCurrent indicator values:")
    print(f"- RSI(14): {float(df['RSI14'].iloc[-1]):.2f}")
    print(f"- MACD: {float(df['MACD'].iloc[-1]):.4f}")
    print(f"- Stochastic %K: {float(df['%K'].iloc[-1]):.2f}")
    print(f"- Stochastic %D: {float(df['%D'].iloc[-1]):.2f}")
    print(f"- Parabolic SAR: {float(df['PSAR'].iloc[-1]):.2f}")

    print(f"\nFibonacci levels (based on last {FIB_LOOKBACK_DAYS} days):")
    for name, level in fibs.items():
        print(f"- {name}: ${float(level):.2f}")

    return df, fibs

if __name__ == "__main__":
    from data_prep import download_and_prepare_data
    df, r, px, df_raw = download_and_prepare_data()
    df, fibs = compute_technical_indicators(df)
