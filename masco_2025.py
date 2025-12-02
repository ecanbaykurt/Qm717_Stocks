# -*- coding: utf-8 -*-
"""Masco_2025 Analysis Module

This module contains functions for analyzing MASCO stock returns,
including regression analysis, normal distribution fitting, and CDF plotting.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from stargazer.stargazer import Stargazer
from scipy.stats import norm
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
import io
from datetime import datetime

def load_data(stock_ticker="MAS", start="2005-01-01", end="2025-01-30"):
    """Load stock data and calculate monthly returns."""
    tickers = [stock_ticker, "^GSPC", "^W5000", "^TYX"]
    
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False
    )["Adj Close"]
    
    raw.columns = [stock_ticker, "SP500", "VW", "TYX"]
    raw = raw.dropna()
    
    # Convert to monthly prices (end of month)
    monthly_prices = raw.resample("ME").last()
    
    # Calculate log returns
    returns = np.log(monthly_prices / monthly_prices.shift(1)).dropna()
    
    return returns


def run_regressions(returns, stock_ticker="MAS"):
    """Run all regression models and return fitted models."""
    y = returns[stock_ticker]
    
    # Regression 1: MAS ~ SP500
    X1 = sm.add_constant(returns["SP500"])
    reg1 = sm.OLS(y, X1).fit()
    
    # Regression 2: MAS ~ VW
    X2 = sm.add_constant(returns["VW"])
    reg2 = sm.OLS(y, X2).fit()
    
    # Regression 3: MAS ~ TYX
    X3 = sm.add_constant(returns["TYX"])
    reg3 = sm.OLS(y, X3).fit()
    
    # Regression 4: MAS ~ SP500 + VW
    X4 = sm.add_constant(returns[["SP500", "VW"]])
    reg4 = sm.OLS(y, X4).fit()
    
    # Regression 5: MAS ~ SP500 + VW + TYX
    X5 = sm.add_constant(returns[["SP500", "VW", "TYX"]])
    reg5 = sm.OLS(y, X5).fit()
    
    return [reg1, reg2, reg3, reg4, reg5]


def create_regression_table_dataframe(regressions, stock_ticker="MAS", stock_name="MASCO"):
    """Create a DataFrame with regression results in the new format."""
    # Helper function to create cell value with coefficient and standard error
    def create_cell_value(reg, var_name):
        if var_name in reg.params.index:
            val = reg.params[var_name]
            se = reg.bse.get(var_name, 0.0)
            pval = reg.pvalues.get(var_name, 1.0)
            sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
            return f"{val:.4f}{sig} ({se:.4f})"
        else:
            return ""
    
    # Create data for the table
    table_data = []
    for i, reg in enumerate(regressions, 1):
        row = {
            'Regression': f'({i})',
            'Int.': create_cell_value(reg, 'const'),
            'S&P 500': create_cell_value(reg, 'SP500'),
            'Val.-Wgtd': create_cell_value(reg, 'VW'),
            '30 Yr Treas.': create_cell_value(reg, 'TYX'),
        }
        
        # Add S,R² column (Residual Std. Error and R²)
        resid_se = np.sqrt(reg.mse_resid)
        r_squared = reg.rsquared
        row['S,R²'] = f"{resid_se:.4f}, {r_squared:.4f}"
        
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    return df


def create_stargazer_table(regressions, stock_ticker="MAS", stock_name="MASCO"):
    """Create and return Stargazer table HTML with improved styling for visibility."""
    sg = Stargazer(regressions)
    sg.title(f"{stock_name} ({stock_ticker}) Monthly Return Regressions (20 Years)")
    sg.custom_columns(["(1)", "(2)", "(3)", "(4)", "(5)"], [1,1,1,1,1])
    sg.significant_digits(4)
    html_content = sg.render_html()
    
    # Add CSS styling to ensure table is visible with dark text
    styled_html = f"""
    <style>
        .stargazer-table {{
            color: #000000 !important;
            background-color: #ffffff !important;
        }}
        .stargazer-table table {{
            color: #000000 !important;
            background-color: #ffffff !important;
            border-collapse: collapse;
            width: 100%;
        }}
        .stargazer-table th,
        .stargazer-table td {{
            color: #000000 !important;
            background-color: #ffffff !important;
            border: 1px solid #cccccc;
            padding: 8px;
            text-align: left;
        }}
        .stargazer-table th {{
            background-color: #f0f0f0 !important;
            font-weight: bold;
        }}
        .stargazer-table tr:nth-child(even) {{
            background-color: #f9f9f9 !important;
        }}
        .stargazer-table * {{
            color: #000000 !important;
        }}
    </style>
    <div class="stargazer-table" style="background-color: #ffffff; color: #000000; padding: 20px;">
        {html_content}
    </div>
    """
    return styled_html


def plot_normal_distribution(returns, stock_ticker="MAS"):
    """Create normal distribution plot and return figure."""
    stock_returns = returns[stock_ticker].dropna()
    
    # Fit normal distribution
    mu, sigma = norm.fit(stock_returns)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(stock_returns, bins=25, density=True, alpha=0.6, color='skyblue', label=f"{stock_ticker} Monthly Returns")
    
    # Normal distribution curve
    xmin, xmax = ax.get_xlim()
    xx = np.linspace(xmin, xmax, 200)
    yy = norm.pdf(xx, mu, sigma)
    ax.plot(xx, yy, 'r', linewidth=2, label=f"Normal Fit (μ={mu:.4f}, σ={sigma:.4f})")
    
    ax.set_title(f"{stock_ticker} Monthly Returns with Normal Distribution Fit")
    ax.set_xlabel("Monthly Return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    return fig


def plot_cdf(returns, stock_ticker="MAS"):
    """Create CDF plot and return figure."""
    stock_returns = returns[stock_ticker].dropna()
    sp = returns["SP500"].dropna()
    
    # Sort returns
    stock_sorted = np.sort(stock_returns)
    sp_sorted = np.sort(sp)
    
    # Cumulative probabilities
    p_stock = np.linspace(0, 1, len(stock_sorted))
    p_sp = np.linspace(0, 1, len(sp_sorted))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(stock_sorted, p_stock, s=20, marker='^', color="red", alpha=0.7, label=stock_ticker)
    ax.scatter(sp_sorted, p_sp, s=20, marker='s', color="blue", alpha=0.7, label="S&P 500")
    
    ax.set_title(f"Distribution of Monthly Returns: {stock_ticker} vs S&P 500")
    ax.set_xlabel("Return")
    ax.set_ylabel("Cumulative Probability")
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    return fig


def generate_pdf_report(returns, regressions, fig_normal, fig_cdf, start_date, end_date, stock_ticker="MAS", stock_name="MASCO"):
    """Generate a PDF report with all analysis results."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                          rightMargin=0.75*inch, leftMargin=0.75*inch,
                          topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    # Container for the 'Flowable' objects
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    title = Paragraph(f"{stock_name} ({stock_ticker}) Stock Returns Analysis", title_style)
    elements.append(title)
    
    # Subtitle
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        alignment=TA_CENTER,
        spaceAfter=20
    )
    subtitle = Paragraph(f"Analysis Period: {start_date} to {end_date}", subtitle_style)
    elements.append(subtitle)
    elements.append(Spacer(1, 0.2*inch))
    
    # Educational Introduction Section
    intro_style = ParagraphStyle(
        'IntroStyle',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8,
        leftIndent=0
    )
    
    intro_title_style = ParagraphStyle(
        'IntroTitle',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=10
    )
    
    elements.append(Paragraph("Why This Analysis Matters", intro_title_style))
    
    intro_text = [
        "<b>What are we doing here?</b><br/>Think of this like a detective story! We're trying to figure out what makes a stock's price move. Is it the overall market? Interest rates? Something else?",
        "<b>Why should you care?</b><br/>• <b>For Investors:</b> Understanding what drives a stock helps you make smarter investment decisions. If you know a stock moves 1.5x with the market, you can better predict what might happen.<br/>• <b>For Companies:</b> Companies use this to understand their stock's risk profile and how it compares to competitors.<br/>• <b>For Students:</b> This is real-world finance! You're learning the same tools that professionals use every day.",
        "<b>What can we learn from these results?</b><br/>1. <b>Beta (Market Sensitivity):</b> The S&P 500 coefficient tells us how much the stock moves when the market moves. If it's 1.5, the stock moves 1.5% when the market moves 1% (more volatile!). If it's 0.8, the stock moves 0.8% when the market moves 1% (less volatile).<br/>2. <b>Risk Factors:</b> Which factors actually matter? The stars (***, **, *) tell us what's statistically significant.<br/>3. <b>Model Quality:</b> R² tells us how well our model explains the stock's movements. Higher is better!",
        "<b>Real-World Examples:</b><br/>• <b>Tech Stocks</b> (like Apple, Microsoft): Often have high betas (1.2-1.5) - they swing more than the market<br/>• <b>Utility Stocks</b> (like power companies): Usually have low betas (0.5-0.8) - they're more stable<br/>• <b>Gold Stocks:</b> Sometimes have negative correlation with the market - they go up when markets go down!",
        "<b>The Bottom Line:</b> This analysis helps answer: 'If the market goes up 10%, how much will my stock go up?' That's super useful for anyone investing money!"
    ]
    
    for text in intro_text:
        elements.append(Paragraph(text, intro_style))
        elements.append(Spacer(1, 0.15*inch))
    
    elements.append(Spacer(1, 0.2*inch))
    
    # Page 1: Regression Table
    elements.append(Paragraph(f"{stock_name} ({stock_ticker}) Monthly Return Regressions", styles['Heading2']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Create regression table in the new format: Regression | Int. | S&P 500 | Val.-Wgtd | 30 Yr Treas. | S,R²
    # Map variable names to display names
    var_display_map = {
        'const': 'Int.',
        'SP500': 'S&P 500',
        'VW': 'Val.-Wgtd',
        'TYX': '30 Yr Treas.'
    }
    
    # Helper function to create cell with coefficient and standard error
    def create_cell_value(reg, var_name):
        if var_name in reg.params.index:
            val = reg.params[var_name]
            se = reg.bse.get(var_name, 0.0)
            pval = reg.pvalues.get(var_name, 1.0)
            sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
            # Format: coefficient with stars, standard error in parentheses on same line
            return f"{val:.4f}{sig} ({se:.4f})"
        else:
            return ""
    
    # Create table header
    table_data = [['Regression', 'Int.', 'S&P 500', 'Val.-Wgtd', '30 Yr Treas.', 'S,R²']]
    
    # Add rows for each regression
    for i, reg in enumerate(regressions, 1):
        row = [f"({i})"]
        
        # Add intercept
        row.append(create_cell_value(reg, 'const'))
        
        # Add S&P 500
        row.append(create_cell_value(reg, 'SP500'))
        
        # Add Val.-Wgtd (VW)
        row.append(create_cell_value(reg, 'VW'))
        
        # Add 30 Yr Treas. (TYX)
        row.append(create_cell_value(reg, 'TYX'))
        
        # Add S,R² column (Residual Std. Error and R²)
        resid_se = np.sqrt(reg.mse_resid)
        r_squared = reg.rsquared
        row.append(f"{resid_se:.4f}, {r_squared:.4f}")
        
        table_data.append(row)
    
    # Create table with proper column widths
    reg_table = Table(table_data, colWidths=[0.8*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    reg_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    elements.append(reg_table)
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("* p<0.1; ** p<0.05; *** p<0.01", styles['Normal']))
    
    # Add Interpretation Guide
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph("How to Read This Table", intro_title_style))
    
    interpretation_text = [
        "<b>The Numbers:</b>",
        "• <b>Coefficients</b> (the big numbers): Tell you how much the stock moves when that factor moves by 1%",
        "• <b>Standard Errors</b> (in parentheses): Show how precise our estimate is - smaller is better!",
        "• <b>Stars (***, **, *):</b> Tell you if the result is statistically significant (not just random luck)",
        "",
        "<b>What Each Model Tells Us:</b>",
        "• <b>Model (1):</b> How does the stock move with the S&P 500? This is like asking 'Is this stock more or less volatile than the market?'",
        "• <b>Model (2):</b> How does it move with the value-weighted market? This gives a broader market view",
        "• <b>Model (3):</b> How does it react to interest rates? Higher rates usually hurt stocks, but by how much?",
        "• <b>Model (4):</b> Combines S&P 500 and value-weighted - does this help explain more?",
        "• <b>Model (5):</b> The full model - all factors together. Usually has the highest R²!",
        "",
        "<b>The S,R² Column:</b>",
        "• <b>S</b> (first number): Residual standard error - how far off our predictions are on average. Lower is better!",
        "• <b>R²</b> (second number): How much of the stock's movement we can explain. 0.5 means we explain 50% - pretty good!"
    ]
    
    for text in interpretation_text:
        if text == "":
            elements.append(Spacer(1, 0.1*inch))
        else:
            elements.append(Paragraph(text, intro_style))
            elements.append(Spacer(1, 0.08*inch))
    
    # Add Excel Tutorial Section
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph("How to Perform This Analysis in Excel", styles['Heading2']))
    elements.append(Spacer(1, 0.15*inch))
    
    # Excel tutorial content
    excel_instructions = [
        ("<b>Step 1: Prepare Your Data</b>", [
            "1. Download monthly stock price data from Yahoo Finance or other sources",
            "2. Calculate monthly returns using the formula: =LN(Price_t / Price_{t-1})",
            "3. Organize your data in columns: Date, Stock Returns, S&P 500 Returns, Value-Weighted Returns, 30-Year Treasury Returns"
        ]),
        ("<b>Step 2: Install Data Analysis ToolPak</b>", [
            "1. Go to File → Options → Add-ins",
            "2. Select 'Analysis ToolPak' and click 'Go'",
            "3. Check the box for 'Analysis ToolPak' and click 'OK'",
            "4. You should now see 'Data Analysis' in the Data tab"
        ]),
        ("<b>Step 3: Run Regression Analysis</b>", [
            "1. Go to Data → Data Analysis → Regression",
            "2. <b>Input Y Range:</b> Select your stock returns column (dependent variable)",
            "3. <b>Input X Range:</b> Select your independent variables (S&P 500, VW, TYX)",
            "4. Check 'Labels' if your first row contains headers",
            "5. Choose an output range or new worksheet",
            "6. Click 'OK' to run the regression"
        ]),
        ("<b>Step 4: Interpret the Results</b>", [
            "• <b>Coefficients:</b> Found in the 'Coefficients' column - these show the relationship strength",
            "• <b>Standard Error:</b> Found in the 'Standard Error' column - measures coefficient precision",
            "• <b>P-value:</b> Found in the 'P-value' column - indicates statistical significance",
            "• <b>R-squared:</b> Found in 'Regression Statistics' - shows how well the model fits (0 to 1)",
            "• <b>Residual Standard Error:</b> Found in 'Regression Statistics' - measures prediction accuracy"
        ]),
        ("<b>Step 5: Multiple Regressions</b>", [
            "To run different models (like in the table above):",
            "• <b>Model (1):</b> Y = Stock Returns, X = S&P 500 only",
            "• <b>Model (2):</b> Y = Stock Returns, X = Value-Weighted only",
            "• <b>Model (3):</b> Y = Stock Returns, X = 30-Year Treasury only",
            "• <b>Model (4):</b> Y = Stock Returns, X = S&P 500 + Value-Weighted",
            "• <b>Model (5):</b> Y = Stock Returns, X = S&P 500 + Value-Weighted + 30-Year Treasury",
            "Run each regression separately and compile results into a table"
        ]),
        ("<b>Step 6: Significance Testing</b>", [
            "• If P-value < 0.01: Highly significant (***)",
            "• If P-value < 0.05: Significant (**)",
            "• If P-value < 0.10: Marginally significant (*)",
            "• If P-value ≥ 0.10: Not significant"
        ])
    ]
    
    # Create styled paragraphs for instructions
    instruction_style = ParagraphStyle(
        'InstructionStyle',
        parent=styles['Normal'],
        fontSize=9,
        leftIndent=0,
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=styles['Normal'],
        fontSize=9,
        leftIndent=20,
        spaceAfter=4,
        bulletIndent=10
    )
    
    for title, items in excel_instructions:
        elements.append(Paragraph(title, instruction_style))
        elements.append(Spacer(1, 0.1*inch))
        for item in items:
            elements.append(Paragraph(f"• {item}", bullet_style))
        elements.append(Spacer(1, 0.15*inch))
    
    # Page break
    elements.append(PageBreak())
    
    # Page 2: Normal Distribution
    elements.append(Paragraph("Normal Distribution Fit", styles['Heading2']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Convert matplotlib figure to image
    img_buffer = io.BytesIO()
    fig_normal.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_normal = Image(img_buffer, width=6.5*inch, height=4*inch)
    elements.append(img_normal)
    
    # Add distribution parameters
    stock_returns = returns[stock_ticker].dropna()
    mu, sigma = norm.fit(stock_returns)
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph(f"Mean (μ): {mu:.4f}", styles['Normal']))
    elements.append(Paragraph(f"Standard Deviation (σ): {sigma:.4f}", styles['Normal']))
    elements.append(Paragraph(f"Variance (σ²): {sigma**2:.4f}", styles['Normal']))
    
    # Add Excel tutorial for Normal Distribution
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph("How to Create Normal Distribution Graph in Excel", styles['Heading2']))
    elements.append(Spacer(1, 0.15*inch))
    
    normal_dist_instructions = [
        ("<b>Step 1: Calculate Statistics</b>", [
            "1. Calculate mean: =AVERAGE(returns_range)",
            "2. Calculate standard deviation: =STDEV.S(returns_range)",
            "3. Create bins for histogram: Create a column with bin ranges (e.g., -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15)"
        ]),
        ("<b>Step 2: Create Histogram</b>", [
            "1. Select your returns data",
            "2. Go to Insert → Charts → Histogram (or use Data Analysis → Histogram)",
            "3. If using Data Analysis:",
            "   • Input Range: Select your returns data",
            "   • Bin Range: Select your bin ranges",
            "   • Check 'Chart Output'",
            "4. Format the histogram: Right-click → Format Data Series → Adjust gap width"
        ]),
        ("<b>Step 3: Add Normal Distribution Curve</b>", [
            "1. Create a new column for normal distribution values",
            "2. Use formula: =NORM.DIST(x, mean, std_dev, FALSE) where:",
            "   • x = bin value",
            "   • mean = calculated mean from Step 1",
            "   • std_dev = calculated standard deviation from Step 1",
            "   • FALSE = returns probability density (not cumulative)",
            "3. Create a scatter plot with your bin values and normal distribution values",
            "4. Add this as a line series to your histogram chart",
            "5. Right-click chart → Select Data → Add → Select normal distribution data"
        ]),
        ("<b>Step 4: Format the Chart</b>", [
            "1. Add chart title: 'Stock Returns with Normal Distribution Fit'",
            "2. Label axes: X-axis = 'Monthly Return', Y-axis = 'Density'",
            "3. Add legend to distinguish histogram from normal curve",
            "4. Format histogram bars: Set transparency (alpha) to ~60%",
            "5. Format normal curve: Use a different color (e.g., red) with thicker line"
        ])
    ]
    
    for title, items in normal_dist_instructions:
        elements.append(Paragraph(title, instruction_style))
        elements.append(Spacer(1, 0.1*inch))
        for item in items:
            elements.append(Paragraph(f"• {item}", bullet_style))
        elements.append(Spacer(1, 0.15*inch))
    
    # Page break
    elements.append(PageBreak())
    
    # Page 3: CDF Graph
    elements.append(Paragraph("Cumulative Distribution Function (CDF)", styles['Heading2']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Convert CDF figure to image
    img_buffer2 = io.BytesIO()
    fig_cdf.savefig(img_buffer2, format='png', dpi=150, bbox_inches='tight')
    img_buffer2.seek(0)
    img_cdf = Image(img_buffer2, width=6.5*inch, height=4*inch)
    elements.append(img_cdf)
    
    # Add comparison statistics
    elements.append(Spacer(1, 0.2*inch))
    sp = returns["SP500"].dropna()
    stats_text = f"""
    <b>{stock_ticker} Statistics:</b><br/>
    Mean: {stock_returns.mean():.4f}, Std Dev: {stock_returns.std():.4f}, Min: {stock_returns.min():.4f}, Max: {stock_returns.max():.4f}<br/><br/>
    <b>S&P 500 Statistics:</b><br/>
    Mean: {sp.mean():.4f}, Std Dev: {sp.std():.4f}, Min: {sp.min():.4f}, Max: {sp.max():.4f}
    """
    elements.append(Paragraph(stats_text, styles['Normal']))
    
    # Add Excel tutorial for CDF Graph
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph("How to Create CDF Graph in Excel", styles['Heading2']))
    elements.append(Spacer(1, 0.15*inch))
    
    cdf_instructions = [
        ("<b>Step 1: Prepare Data</b>", [
            "1. Sort your returns data in descending order: Select data → Data → Sort → Largest to Smallest",
            "2. Create a column for cumulative probability",
            "3. Use formula: =1-ROW()/COUNT($A$2:$A$N) where N is your last row",
            "   Or use: =1-(ROW()-1)/(COUNT(A:A)-1) if starting from row 2",
            "4. This creates values from 0 to 1 representing cumulative probability"
        ]),
        ("<b>Step 2: Create Scatter Plot</b>", [
            "1. Select both columns: sorted returns and cumulative probability",
            "2. Go to Insert → Charts → Scatter → Scatter with Markers",
            "3. Excel will create a scatter plot with your data points"
        ]),
        ("<b>Step 3: Add Second Series (for Comparison)</b>", [
            "1. Right-click on the chart → Select Data",
            "2. Click 'Add' to add a new series",
            "3. Series name: 'S&P 500' (or your comparison stock)",
            "4. X values: Select sorted S&P 500 returns",
            "5. Y values: Select cumulative probability for S&P 500",
            "6. Click OK to add the series"
        ]),
        ("<b>Step 4: Format the Chart</b>", [
            "1. Change marker styles:",
            "   • Right-click first series → Format Data Series → Marker Options",
            "   • Choose different marker (e.g., triangles ^ for Stock)",
            "   • Right-click second series → Choose different marker (e.g., squares for S&P 500)",
            "2. Change colors:",
            "   • Format Data Series → Marker Fill → Choose colors (e.g., red for Stock, blue for S&P 500)",
            "3. Add chart title: 'Distribution of Monthly Returns: Stock vs S&P 500'",
            "4. Label axes:",
            "   • X-axis: 'Return'",
            "   • Y-axis: 'Cumulative Probability'",
            "5. Add legend: Chart Tools → Add Chart Element → Legend"
        ]),
        ("<b>Step 5: Adjust Transparency</b>", [
            "1. Right-click each data series → Format Data Series",
            "2. Go to Marker Fill → Transparency",
            "3. Set transparency to ~30% (0.3) for better visibility",
            "4. Adjust marker size if needed: Marker Options → Size"
        ]),
        ("<b>Step 6: Add Gridlines (Optional)</b>", [
            "1. Right-click chart → Add Chart Element → Gridlines",
            "2. Choose Primary Major Horizontal and/or Vertical gridlines",
            "3. This helps with reading values from the chart"
        ])
    ]
    
    for title, items in cdf_instructions:
        elements.append(Paragraph(title, instruction_style))
        elements.append(Spacer(1, 0.1*inch))
        for item in items:
            elements.append(Paragraph(f"• {item}", bullet_style))
        elements.append(Spacer(1, 0.15*inch))
    
    # Footer
    elements.append(Spacer(1, 0.5*inch))
    footer = Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data Source: Yahoo Finance", 
                      ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER))
    elements.append(footer)
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer