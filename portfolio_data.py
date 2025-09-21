"""
Portfolio Data Management Module

This module handles data download, preparation, and basic statistics calculation
for portfolio optimization analysis.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


class PortfolioDataManager:
    """Manages portfolio data download and preparation"""
    
    def __init__(self, years_back=5, risk_free_rate=0.02):
        self.years_back = years_back
        self.risk_free_rate = risk_free_rate
        self.assets = self._get_default_assets()
        self.data = {}
        self.df = None
        self.returns = None
        
    def _get_default_assets(self):
        """Define default asset universe"""
        return {
            # Equity ETFs (Diversified)
            'SPY': 'S&P 500 ETF',
            'QQQ': 'NASDAQ 100 ETF', 
            'EFA': 'EAFE International ETF',
            'EEM': 'Emerging Markets ETF',
            'VTI': 'Total Stock Market ETF',
            
            # Fixed Income (Bonds)
            'TLT': '20+ Year Treasury Bond ETF',
            'IEF': '7-10 Year Treasury Bond ETF',
            'TIP': 'TIPS (Inflation-Protected Securities)',
            'HYG': 'High Yield Corporate Bond ETF',
            'LQD': 'Investment Grade Corporate Bond ETF',
            
            # Real Estate
            'VNQ': 'Real Estate Investment Trust ETF',
            'IYR': 'US Real Estate ETF',
            
            # Commodities
            'GLD': 'Gold ETF',
            'SLV': 'Silver ETF',
            'DJP': 'Commodity ETF'
        }
    
    def download_data(self, custom_assets=None):
        """Download market data for all assets"""
        if custom_assets:
            self.assets = custom_assets
            
        start_date = (datetime.today() - timedelta(days=int(self.years_back*365.25))).strftime("%Y-%m-%d")
        end_date = datetime.today().strftime("%Y-%m-%d")
        
        print(f"Downloading data from {start_date} to {end_date}...")
        
        for ticker, description in self.assets.items():
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if len(df) > 0:
                    # Handle multi-level columns from yfinance
                    if isinstance(df.columns, pd.MultiIndex):
                        close_data = df[('Close', ticker)]
                    else:
                        close_data = df['Close']
                    self.data[ticker] = close_data
                    print(f"✅ {ticker}: {description} - {len(df)} days")
                else:
                    print(f"❌ {ticker}: No data available")
            except Exception as e:
                print(f"❌ {ticker}: Error - {str(e)}")
        
        print(f"\nSuccessfully downloaded {len(self.data)} assets")
        return len(self.data) > 0
    
    def prepare_data(self):
        """Prepare data for analysis"""
        if len(self.data) == 0:
            print("❌ No data available for analysis")
            return False
            
        # Create DataFrame with all assets
        self.df = pd.DataFrame(self.data)
        self.df = self.df.dropna()  # Remove any missing data
        
        # Calculate daily returns
        self.returns = self.df.pct_change().dropna()
        
        print(f"Data prepared: {len(self.df)} days, {len(self.df.columns)} assets")
        print(f"Date range: {self.df.index[0].strftime('%Y-%m-%d')} to {self.df.index[-1].strftime('%Y-%m-%d')}")
        
        return True
    
    def calculate_statistics(self):
        """Calculate basic statistics for all assets"""
        if self.returns is None:
            print("❌ No returns data available")
            return None
            
        # Calculate annualized statistics
        annual_returns = self.returns.mean() * 252
        annual_volatility = self.returns.std() * np.sqrt(252)
        sharpe_ratios = (annual_returns - self.risk_free_rate) / annual_volatility
        
        # Create summary statistics
        summary_stats = pd.DataFrame({
            'Annual Return': annual_returns,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratios,
            'Max Drawdown': self.returns.min(),
            'Skewness': self.returns.skew(),
            'Kurtosis': self.returns.kurtosis()
        })
        
        return summary_stats
    
    def get_data_summary(self):
        """Get summary of downloaded data"""
        if self.df is None:
            return "No data available"
            
        return {
            'num_assets': len(self.df.columns),
            'num_days': len(self.df),
            'date_range': (self.df.index[0].strftime('%Y-%m-%d'), 
                          self.df.index[-1].strftime('%Y-%m-%d')),
            'assets': list(self.df.columns)
        }


def main():
    """Example usage of PortfolioDataManager"""
    # Create data manager
    data_manager = PortfolioDataManager(years_back=5, risk_free_rate=0.02)
    
    # Download data
    if data_manager.download_data():
        # Prepare data
        if data_manager.prepare_data():
            # Calculate statistics
            stats = data_manager.calculate_statistics()
            if stats is not None:
                print("\nAsset Performance Summary:")
                print(stats.round(4))
                
                # Get data summary
                summary = data_manager.get_data_summary()
                print(f"\nData Summary:")
                print(f"- Assets: {summary['num_assets']}")
                print(f"- Days: {summary['num_days']}")
                print(f"- Date Range: {summary['date_range'][0]} to {summary['date_range'][1]}")


if __name__ == "__main__":
    main()
