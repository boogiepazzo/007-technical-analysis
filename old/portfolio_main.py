"""
Portfolio Optimization Main Module

This module integrates all portfolio optimization components and provides
a complete analysis pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Import our custom modules
from portfolio_data import PortfolioDataManager
from portfolio_optimization import PortfolioOptimizer, RiskMetrics
from monte_carlo import MonteCarloSimulator
from portfolio_report import PortfolioReportGenerator


class PortfolioAnalysisPipeline:
    """Complete portfolio optimization analysis pipeline"""
    
    def __init__(self, years_back=5, risk_free_rate=0.02, target_return=0.08):
        self.years_back = years_back
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        
        # Initialize components
        self.data_manager = PortfolioDataManager(years_back, risk_free_rate)
        self.optimizer = None
        self.monte_carlo = None
        self.report_generator = PortfolioReportGenerator()
        
        # Results storage
        self.results = {
            'data_summary': None,
            'portfolio_analysis': None,
            'optimal_portfolios': None,
            'monte_carlo_results': None,
            'efficient_frontier': None
        }
    
    def run_complete_analysis(self, custom_assets=None, generate_pdf=True):
        """Run complete portfolio optimization analysis"""
        print("🚀 Starting Complete Portfolio Optimization Analysis")
        print("="*60)
        
        # Step 1: Data Management
        print("\n📊 Step 1: Data Download and Preparation")
        if not self.data_manager.download_data(custom_assets):
            print("❌ Failed to download data")
            return False
        
        if not self.data_manager.prepare_data():
            print("❌ Failed to prepare data")
            return False
        
        # Step 2: Portfolio Optimization
        print("\n🎯 Step 2: Portfolio Optimization")
        self.optimizer = PortfolioOptimizer(self.data_manager.returns, self.risk_free_rate)
        
        # Generate efficient frontier
        self.results['efficient_frontier'] = self.optimizer.generate_efficient_frontier()
        
        # Find optimal portfolios
        self.results['optimal_portfolios'] = self.optimizer.find_optimal_portfolios()
        
        # Add target return portfolio if feasible
        target_portfolio = self.optimizer.find_target_return_portfolio(self.target_return)
        if target_portfolio:
            self.results['optimal_portfolios']['Target Return'] = target_portfolio
        
        # Step 3: Risk Analysis
        print("\n⚠️ Step 3: Risk Analysis")
        self._perform_risk_analysis()
        
        # Step 4: Monte Carlo Simulation
        print("\n🎲 Step 4: Monte Carlo Simulation")
        self.monte_carlo = MonteCarloSimulator(self.data_manager.returns, self.risk_free_rate)
        self.results['monte_carlo_results'] = self.monte_carlo.simulate_multiple_portfolios(
            self.results['optimal_portfolios']
        )
        
        # Step 5: Visualization
        print("\n📈 Step 5: Visualization")
        self._create_visualizations()
        
        # Step 6: Report Generation
        if generate_pdf:
            print("\n📄 Step 6: PDF Report Generation")
            self._generate_pdf_report()
        
        print("\n✅ Complete analysis finished successfully!")
        return True
    
    def _perform_risk_analysis(self):
        """Perform comprehensive risk analysis"""
        portfolio_analysis = {}
        
        for name, portfolio in self.results['optimal_portfolios'].items():
            weights = portfolio['weights']
            portfolio_returns = np.dot(self.data_manager.returns, weights)
            
            # Calculate comprehensive risk metrics
            annual_return, annual_vol, sharpe = self.optimizer.portfolio_performance(weights)
            risk_metrics = RiskMetrics.calculate_portfolio_risk_metrics(weights, self.data_manager.returns)
            
            portfolio_analysis[name] = {
                'Annual Return': annual_return,
                'Annual Volatility': annual_vol,
                'Sharpe Ratio': sharpe,
                **risk_metrics,
                'Weights': weights
            }
        
        self.results['portfolio_analysis'] = portfolio_analysis
    
    def _create_visualizations(self):
        """Create comprehensive visualizations"""
        # Set up plotting style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (16, 10)
        
        # Efficient Frontier Plot
        self._plot_efficient_frontier()
        
        # Portfolio Weights Comparison
        self._plot_portfolio_weights()
        
        # Risk-Return Analysis
        self._plot_risk_return_analysis()
        
        # Monte Carlo Visualizations
        self.monte_carlo.create_visualizations(self.results['monte_carlo_results'])
    
    def _plot_efficient_frontier(self):
        """Plot efficient frontier"""
        plt.figure(figsize=(12, 8))
        
        # Plot individual assets
        annual_returns = self.data_manager.returns.mean() * 252
        annual_volatility = self.data_manager.returns.std() * np.sqrt(252)
        plt.scatter(annual_volatility, annual_returns, alpha=0.6, s=50, label='Individual Assets')
        
        # Plot efficient frontier
        efficient_portfolios = self.results['efficient_frontier']
        plt.plot(efficient_portfolios['volatility'], efficient_portfolios['return'], 
                'b-', linewidth=2, label='Efficient Frontier')
        
        # Mark optimal portfolios
        colors = ['red', 'green', 'purple', 'orange']
        for i, (name, portfolio) in enumerate(self.results['optimal_portfolios'].items()):
            plt.scatter(portfolio['volatility'], portfolio['return'], 
                       color=colors[i % len(colors)], s=100, marker='*', label=name)
        
        plt.xlabel('Volatility (Risk)')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def _plot_portfolio_weights(self):
        """Plot portfolio weights comparison"""
        plt.figure(figsize=(14, 8))
        
        weights_df = pd.DataFrame({
            name: portfolio['weights'] 
            for name, portfolio in self.results['optimal_portfolios'].items()
        }, index=self.data_manager.returns.columns)
        
        weights_df.plot(kind='bar', width=0.8)
        plt.title('Portfolio Weights Comparison')
        plt.xlabel('Assets')
        plt.ylabel('Weight')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def _plot_risk_return_analysis(self):
        """Plot risk-return analysis"""
        plt.figure(figsize=(12, 8))
        
        analysis_df = pd.DataFrame({
            name: {k: v for k, v in metrics.items() if k != 'Weights'}
            for name, metrics in self.results['portfolio_analysis'].items()
        }).T
        
        plt.scatter(analysis_df['Annual Volatility'], analysis_df['Annual Return'], 
                   s=100, alpha=0.7, c=analysis_df['Sharpe Ratio'], cmap='viridis')
        
        # Add labels for each portfolio
        for i, name in enumerate(analysis_df.index):
            plt.annotate(name, (analysis_df.iloc[i]['Annual Volatility'], 
                              analysis_df.iloc[i]['Annual Return']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.title('Portfolio Risk-Return Profile')
        plt.colorbar(label='Sharpe Ratio')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def _generate_pdf_report(self):
        """Generate PDF report"""
        data_summary = {
            'num_assets': len(self.data_manager.returns.columns),
            'num_days': len(self.data_manager.df),
            'years_back': self.years_back,
            'risk_free_rate': self.risk_free_rate,
            'target_return': self.target_return,
            'date_range': (self.data_manager.df.index[0].strftime('%Y-%m-%d'), 
                          self.data_manager.df.index[-1].strftime('%Y-%m-%d'))
        }
        
        success = self.report_generator.generate_report(
            data_summary=data_summary,
            portfolio_analysis=self.results['portfolio_analysis'],
            monte_carlo_results=self.results['monte_carlo_results'],
            efficient_frontier_data=self.results['efficient_frontier'],
            create_subfolder=True
        )
        
        if success:
            print("✅ PDF report generated successfully!")
        else:
            print("❌ Failed to generate PDF report")
    
    def get_summary(self):
        """Get analysis summary"""
        if not self.results['portfolio_analysis']:
            return "No analysis results available"
        
        analysis_df = pd.DataFrame({
            name: {k: v for k, v in metrics.items() if k != 'Weights'}
            for name, metrics in self.results['portfolio_analysis'].items()
        }).T
        
        return {
            'best_sharpe': analysis_df['Sharpe Ratio'].idxmax(),
            'best_volatility': analysis_df['Annual Volatility'].idxmin(),
            'best_return': analysis_df['Annual Return'].idxmax(),
            'analysis_summary': analysis_df
        }


def main():
    """Example usage of the complete pipeline"""
    print("🚀 Portfolio Optimization Analysis Pipeline")
    print("="*50)
    
    # Create analysis pipeline
    pipeline = PortfolioAnalysisPipeline(
        years_back=5,
        risk_free_rate=0.02,
        target_return=0.08
    )
    
    # Run complete analysis
    success = pipeline.run_complete_analysis(generate_pdf=True)
    
    if success:
        # Get summary
        summary = pipeline.get_summary()
        print(f"\n📊 Analysis Summary:")
        print(f"Best Sharpe Ratio Portfolio: {summary['best_sharpe']}")
        print(f"Best Volatility Portfolio: {summary['best_volatility']}")
        print(f"Best Return Portfolio: {summary['best_return']}")
        
        print("\n✅ Complete analysis pipeline executed successfully!")
    else:
        print("❌ Analysis pipeline failed")


if __name__ == "__main__":
    main()
