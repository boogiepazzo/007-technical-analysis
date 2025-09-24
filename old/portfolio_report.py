"""
Portfolio Report Generation Module

This module generates comprehensive PDF reports for portfolio optimization analysis.
"""

import os
from datetime import datetime
from typing import Dict, Any
import warnings
warnings.filterwarnings("ignore")

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class PortfolioReportGenerator:
    """Generate comprehensive PDF reports for portfolio analysis"""
    
    def __init__(self, output_dir="."):
        self.output_dir = output_dir
        self.styles = None
        self.title_style = None
        self.heading_style = None
        
        if REPORTLAB_AVAILABLE:
            self._setup_styles()
    
    def _setup_styles(self):
        """Setup report styles"""
        self.styles = getSampleStyleSheet()
        
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue
        )
    
    def generate_report(self, data_summary, portfolio_analysis, monte_carlo_results, 
                       efficient_frontier_data=None, filename=None, create_subfolder=True):
        """Generate comprehensive PDF report"""
        if not REPORTLAB_AVAILABLE:
            print("❌ ReportLab not available. Please install: pip install reportlab")
            return False
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"portfolio_optimization_report_{timestamp}.pdf"
        
        # Create subfolder if requested
        if create_subfolder:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            reports_folder = f"portfolio_reports_{timestamp}"
            if not os.path.exists(reports_folder):
                os.makedirs(reports_folder)
                print(f"📁 Created reports folder: {reports_folder}")
            filepath = os.path.join(reports_folder, filename)
            self.output_dir = reports_folder  # Update output directory
        else:
            filepath = os.path.join(self.output_dir, filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(filepath, pagesize=A4, 
                              rightMargin=72, leftMargin=72, 
                              topMargin=72, bottomMargin=18)
        
        # Build PDF content
        story = []
        
        # Title
        story.append(Paragraph("Portfolio Optimization Analysis Report", self.title_style))
        story.append(Spacer(1, 12))
        
        # Executive Summary
        story.extend(self._create_executive_summary(data_summary))
        
        # Portfolio Performance Summary
        story.extend(self._create_performance_summary(portfolio_analysis))
        
        # Monte Carlo Results
        story.extend(self._create_monte_carlo_summary(monte_carlo_results))
        
        # Investment Recommendations
        story.extend(self._create_recommendations(portfolio_analysis))
        
        # Risk Warnings
        story.extend(self._create_risk_warnings())
        
        # Footer
        story.extend(self._create_footer())
        
        # Build PDF
        try:
            doc.build(story)
            print(f"✅ PDF report generated successfully: {filepath}")
            print(f"📄 Report size: {os.path.getsize(filepath) / 1024:.1f} KB")
            
            # Save additional data files if subfolder was created
            if create_subfolder:
                self._save_additional_data_files(data_summary, portfolio_analysis, monte_carlo_results, efficient_frontier_data)
            
            return True
        except Exception as e:
            print(f"❌ Error generating PDF: {str(e)}")
            return False
    
    def _create_executive_summary(self, data_summary):
        """Create executive summary section"""
        story = []
        story.append(Paragraph("Executive Summary", self.heading_style))
        
        summary_text = f"""
        This report presents a comprehensive portfolio optimization analysis using modern portfolio theory, 
        efficient frontier analysis, and Monte Carlo simulation. The analysis covers {data_summary.get('num_assets', 'N/A')} diversified 
        assets across multiple asset classes and provides optimal portfolio recommendations for different 
        risk tolerance levels.
        
        <b>Key Findings:</b><br/>
        • Analysis Period: {data_summary.get('years_back', 'N/A')} years<br/>
        • Number of Assets Analyzed: {data_summary.get('num_assets', 'N/A')}<br/>
        • Risk-Free Rate: {data_summary.get('risk_free_rate', 0.02)*100:.1f}%<br/>
        • Target Return: {data_summary.get('target_return', 0.08)*100:.1f}%<br/>
        • Monte Carlo Simulations: 10,000 runs per portfolio<br/>
        • Date Range: {data_summary.get('date_range', ['N/A', 'N/A'])[0]} to {data_summary.get('date_range', ['N/A', 'N/A'])[1]}<br/>
        """
        
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 12))
        return story
    
    def _create_performance_summary(self, portfolio_analysis):
        """Create portfolio performance summary section"""
        story = []
        story.append(Paragraph("Portfolio Performance Summary", self.heading_style))
        
        # Create performance table
        performance_data = [['Portfolio', 'Annual Return (%)', 'Volatility (%)', 'Sharpe Ratio', 'VaR (95%)', 'CVaR (95%)']]
        
        for name, metrics in portfolio_analysis.items():
            performance_data.append([
                name,
                f"{metrics['Annual Return']*100:.2f}",
                f"{metrics['Annual Volatility']*100:.2f}",
                f"{metrics['Sharpe Ratio']:.3f}",
                f"{metrics['VaR (95%)']*100:.2f}",
                f"{metrics['CVaR (95%)']*100:.2f}"
            ])
        
        performance_table = Table(performance_data)
        performance_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(performance_table)
        story.append(Spacer(1, 12))
        return story
    
    def _create_monte_carlo_summary(self, monte_carlo_results):
        """Create Monte Carlo simulation summary section"""
        story = []
        story.append(Paragraph("Monte Carlo Simulation Results", self.heading_style))
        story.append(Paragraph("Monte Carlo simulation results for 1-year investment horizon (10,000 simulations per portfolio):", self.styles['Normal']))
        
        # Monte Carlo table
        mc_data = [['Portfolio', 'Mean Value', 'Median Value', '5% VaR', '95% VaR', 'Loss Probability']]
        
        for name, data in monte_carlo_results.items():
            analysis = data['analysis']
            mc_data.append([
                name,
                f"{analysis['mean_final_value']:.4f}",
                f"{analysis['median_final_value']:.4f}",
                f"{analysis['percentiles']['5%']:.4f}",
                f"{analysis['percentiles']['95%']:.4f}",
                f"{analysis['probability_of_loss']:.3f}"
            ])
        
        mc_table = Table(mc_data)
        mc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(mc_table)
        story.append(Spacer(1, 12))
        return story
    
    def _create_recommendations(self, portfolio_analysis):
        """Create investment recommendations section"""
        story = []
        story.append(Paragraph("Investment Recommendations", self.heading_style))
        
        # Find best portfolios
        analysis_df = self._portfolio_analysis_to_df(portfolio_analysis)
        best_sharpe = analysis_df['Sharpe Ratio'].idxmax()
        best_volatility = analysis_df['Annual Volatility'].idxmin()
        best_lpm = analysis_df['LPM (Order 2)'].idxmin()
        
        recommendations_text = f"""
        <b>Recommended Portfolios:</b><br/><br/>
        
        <b>1. Conservative Investors (Low Risk Tolerance):</b><br/>
        → Portfolio: {best_volatility}<br/>
        → Expected Return: {analysis_df.loc[best_volatility, 'Annual Return']*100:.1f}%<br/>
        → Risk Level: {analysis_df.loc[best_volatility, 'Annual Volatility']*100:.1f}%<br/>
        → Sharpe Ratio: {analysis_df.loc[best_volatility, 'Sharpe Ratio']:.3f}<br/><br/>
        
        <b>2. Balanced Investors (Moderate Risk Tolerance):</b><br/>
        → Portfolio: {best_sharpe}<br/>
        → Expected Return: {analysis_df.loc[best_sharpe, 'Annual Return']*100:.1f}%<br/>
        → Risk Level: {analysis_df.loc[best_sharpe, 'Annual Volatility']*100:.1f}%<br/>
        → Sharpe Ratio: {analysis_df.loc[best_sharpe, 'Sharpe Ratio']:.3f}<br/><br/>
        
        <b>3. Risk-Aware Investors (Focus on Downside Protection):</b><br/>
        → Portfolio: {best_lpm}<br/>
        → Expected Return: {analysis_df.loc[best_lpm, 'Annual Return']*100:.1f}%<br/>
        → Risk Level: {analysis_df.loc[best_lpm, 'Annual Volatility']*100:.1f}%<br/>
        → LPM (Downside Risk): {analysis_df.loc[best_lpm, 'LPM (Order 2)']:.6f}<br/>
        """
        
        story.append(Paragraph(recommendations_text, self.styles['Normal']))
        story.append(Spacer(1, 12))
        return story
    
    def _create_risk_warnings(self):
        """Create risk warnings section"""
        story = []
        story.append(Paragraph("Risk Warnings and Disclaimers", self.heading_style))
        
        warnings_text = """
        <b>Important Risk Disclaimers:</b><br/><br/>
        
        • Past performance does not guarantee future results<br/>
        • Market conditions can change rapidly and unpredictably<br/>
        • Diversification does not eliminate all investment risk<br/>
        • All investments carry risk of loss<br/>
        • Consider your personal risk tolerance and investment objectives<br/>
        • Consult with a qualified financial advisor before making investment decisions<br/>
        • This analysis is for educational purposes only and not investment advice<br/>
        • Monte Carlo simulations are based on historical data and normal distribution assumptions<br/>
        • Actual results may vary significantly from simulated results<br/>
        """
        
        story.append(Paragraph(warnings_text, self.styles['Normal']))
        story.append(Spacer(1, 12))
        return story
    
    def _create_footer(self):
        """Create footer section"""
        story = []
        story.append(Paragraph(f"""
        <i>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i><br/>
        <i>Portfolio Optimization Analysis - Modern Portfolio Theory Implementation</i>
        """, self.styles['Normal']))
        return story
    
    def _portfolio_analysis_to_df(self, portfolio_analysis):
        """Convert portfolio analysis to DataFrame for easier manipulation"""
        import pandas as pd
        
        analysis_data = {}
        for name, metrics in portfolio_analysis.items():
            analysis_data[name] = {k: v for k, v in metrics.items() if k != 'Weights'}
        
        return pd.DataFrame(analysis_data).T
    
    def _save_additional_data_files(self, data_summary, portfolio_analysis, monte_carlo_results, efficient_frontier_data):
        """Save additional data files in the reports folder"""
        import pandas as pd
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n💾 Saving additional data files in {self.output_dir}...")
        
        # Save portfolio analysis as CSV
        analysis_df = self._portfolio_analysis_to_df(portfolio_analysis)
        analysis_csv_path = os.path.join(self.output_dir, f"portfolio_analysis_{timestamp}.csv")
        analysis_df.to_csv(analysis_csv_path)
        print(f"✅ Portfolio analysis saved: {analysis_csv_path}")
        
        # Save Monte Carlo results summary
        if monte_carlo_results:
            mc_summary_path = os.path.join(self.output_dir, f"monte_carlo_summary_{timestamp}.csv")
            mc_summary_data = []
            for name, data in monte_carlo_results.items():
                analysis = data['analysis']
                mc_summary_data.append({
                    'Portfolio': name,
                    'Mean_Final_Value': analysis['mean_final_value'],
                    'Median_Final_Value': analysis['median_final_value'],
                    'Std_Final_Value': analysis['std_final_value'],
                    'Min_Final_Value': analysis['min_final_value'],
                    'Max_Final_Value': analysis['max_final_value'],
                    'VaR_5_Percent': analysis['percentiles']['5%'],
                    'VaR_95_Percent': analysis['percentiles']['95%'],
                    'Probability_of_Loss': analysis['probability_of_loss'],
                    'Expected_Shortfall': analysis['expected_shortfall']
                })
            
            mc_summary_df = pd.DataFrame(mc_summary_data)
            mc_summary_df.to_csv(mc_summary_path, index=False)
            print(f"✅ Monte Carlo summary saved: {mc_summary_path}")
        
        # Save efficient frontier data
        if efficient_frontier_data is not None:
            ef_path = os.path.join(self.output_dir, f"efficient_frontier_{timestamp}.csv")
            efficient_frontier_data.to_csv(ef_path, index=False)
            print(f"✅ Efficient frontier data saved: {ef_path}")
        
        # Save run metadata
        metadata_path = os.path.join(self.output_dir, f"run_metadata_{timestamp}.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Portfolio Optimization Analysis Run\n")
            f.write(f"=====================================\n\n")
            f.write(f"Run Timestamp: {timestamp}\n")
            f.write(f"Analysis Period: {data_summary.get('years_back', 'N/A')} years\n")
            f.write(f"Risk-Free Rate: {data_summary.get('risk_free_rate', 0.02)*100:.1f}%\n")
            f.write(f"Target Return: {data_summary.get('target_return', 0.08)*100:.1f}%\n")
            f.write(f"Number of Assets: {data_summary.get('num_assets', 'N/A')}\n")
            f.write(f"Date Range: {data_summary.get('date_range', ['N/A', 'N/A'])[0]} to {data_summary.get('date_range', ['N/A', 'N/A'])[1]}\n")
            f.write(f"Number of Days: {data_summary.get('num_days', 'N/A')}\n")
            f.write(f"Monte Carlo Simulations: 10,000 per portfolio\n")
            f.write(f"Confidence Level: 95%\n\n")
            f.write(f"Assets Analyzed:\n")
            for asset in data_summary.get('assets', []):
                f.write(f"  {asset}\n")
        
        print(f"✅ Run metadata saved: {metadata_path}")
        
        print(f"\n🎯 All files saved in folder: {self.output_dir}")
        print(f"📁 Folder contents:")
        for file in os.listdir(self.output_dir):
            file_path = os.path.join(self.output_dir, file)
            file_size = os.path.getsize(file_path) / 1024
            print(f"   • {file} ({file_size:.1f} KB)")


def install_reportlab():
    """Install ReportLab if not available"""
    import subprocess
    import sys
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
        print("✅ ReportLab installed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error installing ReportLab: {str(e)}")
        return False


def main():
    """Example usage of PortfolioReportGenerator"""
    if not REPORTLAB_AVAILABLE:
        print("ReportLab not available. Installing...")
        if install_reportlab():
            print("Please restart Python to use ReportLab.")
        else:
            print("Failed to install ReportLab.")
    else:
        print("PortfolioReportGenerator module loaded successfully!")
        print("Use with portfolio analysis results to generate PDF reports.")


if __name__ == "__main__":
    main()
