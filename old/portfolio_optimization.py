"""
Portfolio Optimization Module

This module implements modern portfolio theory, efficient frontier analysis,
and various optimization strategies for portfolio construction.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")


class PortfolioOptimizer:
    """Portfolio optimization using modern portfolio theory"""
    
    def __init__(self, returns, risk_free_rate=0.02):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(returns.columns)
        self.efficient_portfolios = None
        
    def portfolio_performance(self, weights):
        """Calculate portfolio performance metrics"""
        portfolio_return = np.sum(weights * self.returns.mean()) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def portfolio_volatility(self, weights):
        """Calculate portfolio volatility"""
        return np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
    
    def portfolio_return(self, weights):
        """Calculate portfolio return"""
        return np.sum(weights * self.returns.mean()) * 252
    
    def negative_sharpe(self, weights):
        """Negative Sharpe ratio for minimization"""
        return -self.portfolio_performance(weights)[2]
    
    def generate_efficient_frontier(self, num_portfolios=50):
        """Generate efficient frontier"""
        annual_returns = self.returns.mean() * 252
        target_returns = np.linspace(annual_returns.min(), annual_returns.max(), num_portfolios)
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            constraints_with_return = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x, target=target_return: self.portfolio_return(x) - target}
            ]
            
            result = minimize(
                self.portfolio_volatility,
                x0=np.array([1/self.num_assets] * self.num_assets),
                args=(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_with_return
            )
            
            if result.success:
                efficient_portfolios.append({
                    'weights': result.x,
                    'return': target_return,
                    'volatility': result.fun,
                    'sharpe': (target_return - self.risk_free_rate) / result.fun
                })
        
        self.efficient_portfolios = pd.DataFrame(efficient_portfolios)
        return self.efficient_portfolios
    
    def find_optimal_portfolios(self):
        """Find optimal portfolios for different objectives"""
        if self.efficient_portfolios is None:
            self.generate_efficient_frontier()
        
        # Maximum Sharpe Ratio Portfolio
        max_sharpe_idx = self.efficient_portfolios['sharpe'].idxmax()
        max_sharpe_portfolio = self.efficient_portfolios.iloc[max_sharpe_idx]
        
        # Minimum Volatility Portfolio
        min_vol_idx = self.efficient_portfolios['volatility'].idxmin()
        min_vol_portfolio = self.efficient_portfolios.iloc[min_vol_idx]
        
        # Risk Parity Portfolio
        risk_parity_weights = self._calculate_risk_parity_weights()
        risk_parity_return, risk_parity_vol, risk_parity_sharpe = self.portfolio_performance(risk_parity_weights)
        
        optimal_portfolios = {
            'Max Sharpe': {
                'weights': max_sharpe_portfolio['weights'],
                'return': max_sharpe_portfolio['return'],
                'volatility': max_sharpe_portfolio['volatility'],
                'sharpe': max_sharpe_portfolio['sharpe']
            },
            'Min Volatility': {
                'weights': min_vol_portfolio['weights'],
                'return': min_vol_portfolio['return'],
                'volatility': min_vol_portfolio['volatility'],
                'sharpe': min_vol_portfolio['sharpe']
            },
            'Risk Parity': {
                'weights': risk_parity_weights,
                'return': risk_parity_return,
                'volatility': risk_parity_vol,
                'sharpe': risk_parity_sharpe
            }
        }
        
        return optimal_portfolios
    
    def _calculate_risk_parity_weights(self):
        """Calculate risk parity weights"""
        cov_matrix = self.returns.cov() * 252
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(self.num_assets)
        weights = np.dot(inv_cov, ones) / np.dot(ones, np.dot(inv_cov, ones))
        return weights
    
    def find_target_return_portfolio(self, target_return):
        """Find portfolio with specific target return"""
        if self.efficient_portfolios is None:
            self.generate_efficient_frontier()
        
        if target_return >= self.efficient_portfolios['return'].min() and \
           target_return <= self.efficient_portfolios['return'].max():
            target_idx = self.efficient_portfolios['return'].sub(target_return).abs().idxmin()
            target_portfolio = self.efficient_portfolios.iloc[target_idx]
            
            return {
                'weights': target_portfolio['weights'],
                'return': target_portfolio['return'],
                'volatility': target_portfolio['volatility'],
                'sharpe': target_portfolio['sharpe']
            }
        else:
            return None


class RiskMetrics:
    """Calculate advanced risk metrics"""
    
    @staticmethod
    def lower_partial_moment(returns, threshold=0, order=2):
        """Calculate Lower Partial Moment (LPM) for downside risk"""
        negative_returns = returns[returns < threshold]
        if len(negative_returns) == 0:
            return 0
        return np.mean(np.power(threshold - negative_returns, order))
    
    @staticmethod
    def portfolio_lpm(weights, returns, threshold=0, order=2):
        """Calculate portfolio LPM"""
        portfolio_returns = np.dot(returns, weights)
        return RiskMetrics.lower_partial_moment(portfolio_returns, threshold, order)
    
    @staticmethod
    def value_at_risk(returns, confidence_level=0.05):
        """Calculate Value at Risk (VaR)"""
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def conditional_value_at_risk(returns, confidence_level=0.05):
        """Calculate Conditional Value at Risk (CVaR)"""
        var = RiskMetrics.value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_portfolio_risk_metrics(weights, returns):
        """Calculate comprehensive risk metrics for a portfolio"""
        portfolio_returns = np.dot(returns, weights)
        
        metrics = {
            'LPM (Order 2)': RiskMetrics.portfolio_lpm(weights, returns, threshold=0, order=2),
            'VaR (95%)': RiskMetrics.value_at_risk(portfolio_returns, 0.05),
            'CVaR (95%)': RiskMetrics.conditional_value_at_risk(portfolio_returns, 0.05),
            'Max Drawdown': portfolio_returns.min(),
            'Skewness': pd.Series(portfolio_returns).skew(),
            'Kurtosis': pd.Series(portfolio_returns).kurtosis()
        }
        
        return metrics


def main():
    """Example usage of PortfolioOptimizer"""
    # This would typically be used with data from portfolio_data.py
    print("PortfolioOptimizer module loaded successfully!")
    print("Use with PortfolioDataManager to perform optimization analysis.")


if __name__ == "__main__":
    main()
