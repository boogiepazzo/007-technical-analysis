"""
Monte Carlo Simulation Module

This module implements Monte Carlo simulation for portfolio risk assessment
and scenario analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")


class MonteCarloSimulator:
    """Monte Carlo simulation for portfolio analysis"""
    
    def __init__(self, returns, risk_free_rate=0.02, random_seed=42):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def simulate_portfolio(self, weights, num_simulations=10000, time_horizon=252):
        """
        Perform Monte Carlo simulation for portfolio returns
        
        Parameters:
        - weights: Portfolio weights
        - num_simulations: Number of simulation runs
        - time_horizon: Investment horizon in days (252 = 1 year)
        
        Returns:
        - Dictionary with simulation results
        """
        # Calculate portfolio statistics
        portfolio_mean = np.sum(weights * self.returns.mean()) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        
        # Generate random returns using normal distribution
        random_returns = np.random.normal(
            portfolio_mean / 252,  # Daily mean
            portfolio_std / np.sqrt(252),  # Daily std
            (num_simulations, time_horizon)
        )
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + random_returns, axis=1)
        final_values = cumulative_returns[:, -1]
        
        return {
            'simulated_returns': random_returns,
            'cumulative_returns': cumulative_returns,
            'final_values': final_values,
            'portfolio_mean': portfolio_mean,
            'portfolio_std': portfolio_std,
            'num_simulations': num_simulations,
            'time_horizon': time_horizon
        }
    
    def analyze_results(self, results, confidence_levels=[0.05, 0.25, 0.5, 0.75, 0.95]):
        """Analyze Monte Carlo simulation results"""
        final_values = results['final_values']
        
        analysis = {
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values),
            'min_final_value': np.min(final_values),
            'max_final_value': np.max(final_values),
            'percentiles': {}
        }
        
        for conf_level in confidence_levels:
            analysis['percentiles'][f'{conf_level*100:.0f}%'] = np.percentile(final_values, conf_level * 100)
        
        # Additional risk metrics
        analysis['probability_of_loss'] = np.mean(final_values < 1.0)
        analysis['probability_of_gain'] = np.mean(final_values > 1.0)
        analysis['expected_shortfall'] = np.mean(final_values[final_values <= analysis['percentiles']['5%']])
        
        return analysis
    
    def simulate_multiple_portfolios(self, portfolios, num_simulations=10000, time_horizon=252):
        """Simulate multiple portfolios"""
        results = {}
        
        for name, portfolio in portfolios.items():
            print(f"Simulating {name} portfolio...")
            # Extract weights from portfolio dictionary
            if isinstance(portfolio, dict):
                weights = portfolio['weights']
            else:
                weights = portfolio
            simulation_results = self.simulate_portfolio(weights, num_simulations, time_horizon)
            analysis = self.analyze_results(simulation_results)
            results[name] = {
                'results': simulation_results,
                'analysis': analysis
            }
        
        return results
    
    def create_visualizations(self, monte_carlo_results, save_plots=False, filename_prefix="monte_carlo", show=True, return_figure=False):
        """Create comprehensive Monte Carlo visualizations.

        Parameters:
        - save_plots: When True, saves a PNG alongside any returned figure
        - filename_prefix: Prefix for saved plot filenames
        - show: When True, displays the figure; set False when compiling PDFs
        - return_figure: When True, returns the matplotlib Figure for external saving
        """
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Monte Carlo Simulation Analysis', fontsize=16, fontweight='bold')
        
        portfolio_names = list(monte_carlo_results.keys())
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
        
        # Plot 1: Distribution of final values
        ax1 = axes[0, 0]
        for i, (name, data) in enumerate(monte_carlo_results.items()):
            ax1.hist(data['results']['final_values'], bins=50, alpha=0.6, 
                    label=name, density=True, color=colors[i % len(colors)])
        ax1.set_xlabel('Final Portfolio Value')
        ax1.set_ylabel('Density')
        ax1.set_title('Final Value Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative probability
        ax2 = axes[0, 1]
        for i, (name, data) in enumerate(monte_carlo_results.items()):
            sorted_values = np.sort(data['results']['final_values'])
            probabilities = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            ax2.plot(sorted_values, probabilities, label=name, linewidth=2, 
                    color=colors[i % len(colors)])
        ax2.set_xlabel('Final Portfolio Value')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Risk-Return scatter with confidence intervals
        ax3 = axes[0, 2]
        for i, (name, data) in enumerate(monte_carlo_results.items()):
            analysis = data['analysis']
            ax3.scatter(analysis['std_final_value'], analysis['mean_final_value'], 
                       s=100, alpha=0.7, label=name, color=colors[i % len(colors)])
            # Add confidence intervals
            ax3.errorbar(analysis['std_final_value'], analysis['mean_final_value'],
                        yerr=analysis['std_final_value'], alpha=0.3, capsize=5,
                        color=colors[i % len(colors)])
        ax3.set_xlabel('Risk (Standard Deviation)')
        ax3.set_ylabel('Expected Return')
        ax3.set_title('Risk-Return Profile')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Value at Risk comparison
        ax4 = axes[1, 0]
        var_data = []
        for name, data in monte_carlo_results.items():
            var_5 = data['analysis']['percentiles']['5%']
            var_data.append(var_5)
        
        bars = ax4.bar(portfolio_names, var_data, alpha=0.7, 
                      color=colors[:len(portfolio_names)])
        ax4.set_xlabel('Portfolio')
        ax4.set_ylabel('5% VaR (Final Value)')
        ax4.set_title('Value at Risk (5%)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, var_data):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 5: Expected Shortfall (CVaR)
        ax5 = axes[1, 1]
        cvar_data = []
        for name, data in monte_carlo_results.items():
            cvar_data.append(data['analysis']['expected_shortfall'])
        
        bars = ax5.bar(portfolio_names, cvar_data, alpha=0.7, 
                      color=colors[:len(portfolio_names)])
        ax5.set_xlabel('Portfolio')
        ax5.set_ylabel('Expected Shortfall (CVaR)')
        ax5.set_title('Expected Shortfall (5%)')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, cvar_data):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 6: Probability of loss
        ax6 = axes[1, 2]
        loss_prob_data = []
        for name, data in monte_carlo_results.items():
            loss_prob_data.append(data['analysis']['probability_of_loss'])
        
        bars = ax6.bar(portfolio_names, loss_prob_data, alpha=0.7, 
                      color=colors[:len(portfolio_names)])
        ax6.set_xlabel('Portfolio')
        ax6.set_ylabel('Probability of Loss')
        ax6.set_title('Probability of Loss')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, loss_prob_data):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{filename_prefix}_analysis.png", dpi=300, bbox_inches='tight')
            print(f"Plots saved as {filename_prefix}_analysis.png")
        
        if show:
            plt.show()
        
        if return_figure:
            return fig
    
    def generate_summary_report(self, monte_carlo_results):
        """Generate a summary report of Monte Carlo results"""
        print("\n" + "="*60)
        print("MONTE CARLO SIMULATION SUMMARY REPORT")
        print("="*60)
        
        for name, data in monte_carlo_results.items():
            analysis = data['analysis']
            print(f"\n{name.upper()} PORTFOLIO:")
            print("-" * 40)
            print(f"Mean Final Value: {analysis['mean_final_value']:.4f}")
            print(f"Median Final Value: {analysis['median_final_value']:.4f}")
            print(f"Standard Deviation: {analysis['std_final_value']:.4f}")
            print(f"Minimum Value: {analysis['min_final_value']:.4f}")
            print(f"Maximum Value: {analysis['max_final_value']:.4f}")
            print(f"Probability of Loss: {analysis['probability_of_loss']:.3f}")
            print(f"Probability of Gain: {analysis['probability_of_gain']:.3f}")
            print(f"Expected Shortfall: {analysis['expected_shortfall']:.4f}")
            
            print("\nPercentiles:")
            for percentile, value in analysis['percentiles'].items():
                print(f"  {percentile}: {value:.4f}")


def main():
    """Example usage of MonteCarloSimulator"""
    print("MonteCarloSimulator module loaded successfully!")
    print("Use with PortfolioOptimizer to perform Monte Carlo analysis.")


if __name__ == "__main__":
    main()
