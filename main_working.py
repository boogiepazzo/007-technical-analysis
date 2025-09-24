# AGNC Technical Analysis - Working Main Script
# This script runs all modules in the correct order

import sys
import os
from datetime import datetime

def main():
    """Main execution function that runs all modules in order"""
    print("="*60)
    print("AGNC TECHNICAL ANALYSIS - MODULAR EXECUTION")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Step 1: Configuration and Imports
        print("Step 1: Loading configuration and imports...")
        import config
        print("✅ Configuration loaded successfully")
        print()
        
        # Step 2: Data Download and Preparation
        print("Step 2: Downloading and preparing data...")
        import data_prep
        df, r, px, df_raw = data_prep.download_and_prepare_data()
        print("✅ Data preparation completed")
        print()
        
        # Step 3: Technical Indicators
        print("Step 3: Computing technical indicators...")
        import technical_indicators
        df, fibs = technical_indicators.compute_technical_indicators(df)
        print("✅ Technical indicators computed")
        print()
        
        # Step 4: Time Series Models
        print("Step 4: Fitting time series models...")
        import time_series_models
        ar_res, arma_res, arima_res, arima_order = time_series_models.fit_time_series_models(r, px)
        print("✅ Time series models fitted")
        print()
        
        # Step 5: Volatility Models
        print("Step 5: Fitting volatility models...")
        import volatility_models
        arch_fit, garch_fit, egarch_fit, egarch_available, resid = volatility_models.fit_volatility_models(arma_res, r)
        vol_forecasts = volatility_models.generate_volatility_forecasts(arch_fit, garch_fit, egarch_fit, egarch_available, r)
        print("✅ Volatility models fitted")
        print()
        
        # Step 6: Forecasting
        print("Step 6: Generating forecasts...")
        import forecasting
        
        return_forecasts = forecasting.generate_return_forecasts(ar_res, arma_res, arima_res, r)
        ensemble_fc, weights = forecasting.create_ensemble_forecast(ar_res, arma_res, arima_res, 
                                                       return_forecasts[0], return_forecasts[1], return_forecasts[2])
        
        mc_paths, mc_percentiles, percentiles = forecasting.run_monte_carlo_simulation(
            return_forecasts[0], return_forecasts[1], return_forecasts[2], 
            vol_forecasts[1], float(df['Close'].iloc[-1]))
        
        historical_risk, forecast_risk = forecasting.calculate_risk_metrics(r, ensemble_fc)
        print("✅ Forecasting completed")
        print()
        
        # Step 7: Setup PDF Generation
        print("Step 7: Setting up PDF generation...")
        import pdf_generation
        doc, pdf_content, add_to_pdf, pdf_filename = pdf_generation.setup_pdf_generation()
        print("✅ PDF generation setup completed")
        print()
        
        # Step 8: Plotting
        print("Step 8: Generating plots...")
        import plotting
        
        plotting.plot_technical_indicators(df, fibs, pdf_content, add_to_pdf)
        plotting.plot_forecasts(r, ar_res, arma_res, arima_res, arima_order, return_forecasts, 
                      ensemble_fc, pdf_content, add_to_pdf)
        plotting.plot_advanced_analysis(df, mc_paths, mc_percentiles, percentiles, garch_fit, 
                             vol_forecasts[1], vol_forecasts[5], vol_forecasts[6], 
                             historical_risk, forecast_risk, weights, ar_res, arma_res, 
                             arima_res, ensemble_fc, egarch_available, vol_forecasts[2], 
                             vol_forecasts[0], pdf_content, add_to_pdf)
        print("✅ All plots generated")
        print()
        
        # Step 9: Generate PDF Report
        print("Step 9: Generating PDF report...")
        pdf_generation.generate_pdf_report(doc, pdf_content, add_to_pdf, pdf_filename, df, fibs, 
                           ar_res, arma_res, arima_res, arima_order, weights, 
                           historical_risk, forecast_risk, garch_fit, vol_forecasts[1], 
                           mc_paths, egarch_available)
        print("✅ PDF report generated")
        print()
        
        # Final Summary
        print("="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"PDF Report: {pdf_filename}")
        print(f"Total plots generated: 17")
        print(f"Analysis includes:")
        print("  ✓ Technical indicators (RSI, MACD, Stochastic, etc.)")
        print("  ✓ Time series models (AR, ARMA, ARIMA)")
        print("  ✓ Volatility models (ARCH, GARCH, EGARCH)")
        print("  ✓ Monte Carlo simulation (1,000+ paths)")
        print("  ✓ Risk metrics (VaR, Expected Shortfall)")
        print("  ✓ Ensemble forecasting with confidence intervals")
        print("  ✓ Professional PDF report with all visualizations")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Analysis failed. Please check the error and try again.")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All modules executed successfully!")
    else:
        print("\n💥 Execution failed!")
        sys.exit(1)
