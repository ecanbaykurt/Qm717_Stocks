import streamlit as st
import pandas as pd
import masco_2025 as masco
from streamlit.components.v1 import html
from scipy.stats import norm

# Import stock tickers - handle import errors gracefully
try:
    import stocks
    if hasattr(stocks, 'stock_tickers'):
        stock_tickers_dict = stocks.stock_tickers
    else:
        # Fallback: define directly if import fails
        stock_tickers_dict = {
            "PCG": "PCG",
            "WABTEC": "WAB",
            "ETR": "ETR",
            "DOV": "DOV",
            "General_Dynamics": "GD",
            "PAR": "PAR",
            "OKE": "OKE",
            "LVS": "LVS",
            "MCO": "MCO",
            "LMT": "LMT",
            "EIX": "EIX",
            "SYK": "SYK",
            "HOLX": "HOLX",
            "MHK": "MHK",
            "NOC": "NOC",
            "IFF": "IFF",
            "AZO": "AZO",
            "Southern_Company": "SO",
            "TTWO": "TTWO",
            "Kimberly_Clark": "KMB",
            "CHD": "CHD",
            "EXR": "EXR",
            "CRL": "CRL",
            "Texas_Instruments": "TXN",
        }
except (ImportError, AttributeError):
    # Fallback: define directly if import fails
    stock_tickers_dict = {
        "PCG": "PCG",
        "WABTEC": "WAB",
        "ETR": "ETR",
        "DOV": "DOV",
        "General_Dynamics": "GD",
        "PAR": "PAR",
        "OKE": "OKE",
        "LVS": "LVS",
        "MCO": "MCO",
        "LMT": "LMT",
        "EIX": "EIX",
        "SYK": "SYK",
        "HOLX": "HOLX",
        "MHK": "MHK",
        "NOC": "NOC",
        "IFF": "IFF",
        "AZO": "AZO",
        "Southern_Company": "SO",
        "TTWO": "TTWO",
        "Kimberly_Clark": "KMB",
        "CHD": "CHD",
        "EXR": "EXR",
        "CRL": "CRL",
        "Texas_Instruments": "TXN",
    }

# Page configuration (will be updated after stock selection)
st.set_page_config(
    page_title="Stock Returns Analysis",
    page_icon="📈",
    layout="wide"
)

# Inject print CSS early
print_css_early = """
<style>
    @media print {
        @page {
            size: letter;
            margin: 0.75in;
        }
        
        /* Hide UI elements */
        header[data-testid="stHeader"],
        [data-testid="stSidebar"],
        [data-testid="stToolbar"],
        .stDeployButton,
        #print-button-container {
            display: none !important;
        }
        
        /* Ensure all content is visible */
        body, html {
            background: white !important;
        }
        
        .main .block-container {
            max-width: 100% !important;
            padding: 0 !important;
        }
        
        .print-page-break-1 {
            page-break-after: always !important;
        }
        
        .element-container {
            page-break-inside: avoid;
        }
        
        [data-testid="stImage"],
        [data-testid="stPyplot"],
        img {
            page-break-inside: avoid !important;
            max-width: 100% !important;
        }
        
        [data-testid="stExpander"] {
            display: none !important;
        }
    }
</style>
"""
html(print_css_early, height=0)

# Sidebar for stock and date range selection
st.sidebar.header("Settings")

# Stock selection
stock_options = {name: ticker for name, ticker in stock_tickers_dict.items()}
# Add MAS as default option
stock_options["MASCO (MAS)"] = "MAS"
stock_options = dict(sorted(stock_options.items()))

selected_stock_name = st.sidebar.selectbox(
    "Select Stock",
    options=list(stock_options.keys()),
    index=list(stock_options.keys()).index("MASCO (MAS)") if "MASCO (MAS)" in stock_options else 0,
    help="Choose a stock to analyze"
)
selected_stock_ticker = stock_options[selected_stock_name]

# Date range selection
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2005-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-01-30"))

# Title and header
st.title(f"📈 {selected_stock_name} Stock Returns Analysis")
st.markdown("### Monthly Return Regressions, Normal Distribution, and CDF Analysis")

# Educational Introduction
with st.expander("🎓 Why This Analysis Matters (Click to Learn More!)", expanded=False):
    st.markdown("""
    **What are we doing here?**
    
    Think of this like a detective story! We're trying to figure out what makes a stock's price move. Is it the overall market? Interest rates? Something else? 
    
    **Why should you care?**
    
    📊 **For Investors:** Understanding what drives a stock helps you make smarter investment decisions. If you know a stock moves 1.5x with the market, you can better predict what might happen.
    
    💼 **For Companies:** Companies use this to understand their stock's risk profile and how it compares to competitors.
    
    🎯 **For Students:** This is real-world finance! You're learning the same tools that professionals use every day.
    
    **What can we learn from these results?**
    
    1. **Beta (Market Sensitivity):** The S&P 500 coefficient tells us how much the stock moves when the market moves. 
       - If it's 1.5, the stock moves 1.5% when the market moves 1% (more volatile!)
       - If it's 0.8, the stock moves 0.8% when the market moves 1% (less volatile)
    
    2. **Risk Factors:** Which factors actually matter? The stars (***, **, *) tell us what's statistically significant.
    
    3. **Model Quality:** R² tells us how well our model explains the stock's movements. Higher is better!
    
    **Real-World Examples:**
    
    - **Tech Stocks (like Apple, Microsoft):** Often have high betas (1.2-1.5) - they swing more than the market
    - **Utility Stocks (like power companies):** Usually have low betas (0.5-0.8) - they're more stable
    - **Gold Stocks:** Sometimes have negative correlation with the market - they go up when markets go down!
    
    **The Bottom Line:**
    
    This analysis helps answer: "If the market goes up 10%, how much will my stock go up?" That's super useful for anyone investing money! 💰
    """)

# Load data with progress indicator
with st.spinner(f"Loading {selected_stock_ticker} stock data..."):
    returns = masco.load_data(
        stock_ticker=selected_stock_ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d")
    )

# Display data info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Number of Observations", returns.shape[0])
with col2:
    st.metric("Number of Variables", returns.shape[1])
with col3:
    st.metric("Date Range", f"{returns.index[0].strftime('%Y-%m')} to {returns.index[-1].strftime('%Y-%m')}")

st.divider()

# Run regressions once for all visualizations
with st.spinner("Running regressions..."):
    regressions = masco.run_regressions(returns, stock_ticker=selected_stock_ticker)
    regression_df = masco.create_regression_table_dataframe(regressions, stock_ticker=selected_stock_ticker, stock_name=selected_stock_name)

# Prepare data for graphs
stock_returns = returns[selected_stock_ticker].dropna()
sp = returns["SP500"].dropna()

# ===== SECTION 1: REGRESSION TABLE =====
st.header(f"📊 {selected_stock_name} Monthly Return Regressions")
st.markdown(f"This table shows regression results for {selected_stock_ticker} returns against various market factors.")

# Display the regression table
st.dataframe(
    regression_df,
    width='stretch',
    hide_index=True
)

# Add significance legend
st.markdown("""
**Note:** Standard errors in parentheses  
\* Significant at 10% level  
\*\* Significant at 5% level  
\*\*\* Significant at 1% level
""")

# Interpretation guide
with st.expander("💡 How to Read This Table (Quick Guide)", expanded=False):
    st.markdown(f"""
    **Let's break down what you're seeing:**
    
    **The Numbers:**
    - **Coefficients** (the big numbers): Tell you how much {selected_stock_ticker} moves when that factor moves by 1%
    - **Standard Errors** (in parentheses): Show how precise our estimate is - smaller is better!
    - **Stars (***, **, *):** Tell you if the result is statistically significant (not just random luck)
    
    **What Each Model Tells Us:**
    - **Model (1):** How does {selected_stock_ticker} move with the S&P 500? This is like asking "Is this stock more or less volatile than the market?"
    - **Model (2):** How does it move with the value-weighted market? This gives a broader market view.
    - **Model (3):** How does it react to interest rates? Higher rates usually hurt stocks, but by how much?
    - **Model (4):** Combines S&P 500 and value-weighted - does this help explain more?
    - **Model (5):** The full model - all factors together. Usually has the highest R²!
    
    **The S,R² Column:**
    - **S (first number):** Residual standard error - how far off our predictions are on average. Lower is better!
    - **R² (second number):** How much of the stock's movement we can explain. 0.5 means we explain 50% - pretty good!
    
    **Quick Example:**
    If Model (1) shows S&P 500 = 1.5***, that means:
    - When the S&P 500 goes up 1%, {selected_stock_ticker} typically goes up 1.5%
    - The *** means this is highly significant (very confident it's not random)
    - This stock is more volatile than the market!
    """)

# Show summary statistics
with st.expander("View Summary Statistics"):
    st.dataframe(returns.describe(), width='stretch')

# Excel tutorial for regression
with st.expander("📘 How to Perform This Analysis in Excel"):
    st.markdown("""
    **Step 1: Prepare Your Data**
    - Download monthly stock price data from Yahoo Finance or other sources
    - Calculate monthly returns using the formula: `=LN(Price_t / Price_{t-1})`
    - Organize your data in columns: Date, Stock Returns, S&P 500 Returns, Value-Weighted Returns, 30-Year Treasury Returns
    
    **Step 2: Install Data Analysis ToolPak**
    - Go to File → Options → Add-ins
    - Select 'Analysis ToolPak' and click 'Go'
    - Check the box for 'Analysis ToolPak' and click 'OK'
    - You should now see 'Data Analysis' in the Data tab
    
    **Step 3: Run Regression Analysis**
    - Go to Data → Data Analysis → Regression
    - **Input Y Range:** Select your stock returns column (dependent variable)
    - **Input X Range:** Select your independent variables (S&P 500, VW, TYX)
    - Check 'Labels' if your first row contains headers
    - Choose an output range or new worksheet
    - Click 'OK' to run the regression
    
    **Step 4: Interpret the Results**
    - **Coefficients:** Found in the 'Coefficients' column - these show the relationship strength
    - **Standard Error:** Found in the 'Standard Error' column - measures coefficient precision
    - **P-value:** Found in the 'P-value' column - indicates statistical significance
    - **R-squared:** Found in 'Regression Statistics' - shows how well the model fits (0 to 1)
    - **Residual Standard Error:** Found in 'Regression Statistics' - measures prediction accuracy
    
    **Step 5: Multiple Regressions**
    To run different models (like in the table above):
    - **Model (1):** Y = Stock Returns, X = S&P 500 only
    - **Model (2):** Y = Stock Returns, X = Value-Weighted only
    - **Model (3):** Y = Stock Returns, X = 30-Year Treasury only
    - **Model (4):** Y = Stock Returns, X = S&P 500 + Value-Weighted
    - **Model (5):** Y = Stock Returns, X = S&P 500 + Value-Weighted + 30-Year Treasury
    Run each regression separately and compile results into a table
    
    **Step 6: Significance Testing**
    - If P-value < 0.01: Highly significant (***)
    - If P-value < 0.05: Significant (**)
    - If P-value < 0.10: Marginally significant (*)
    - If P-value ≥ 0.10: Not significant
    """)

# Page break marker for print
html("""
<div class="print-page-break-1" style="display: none;"></div>
""", height=0)

st.divider()

# ===== SECTION 2: NORMAL DISTRIBUTION =====
st.header("📉 Normal Distribution Fit")
st.markdown(f"Histogram of {selected_stock_ticker} monthly returns with fitted normal distribution curve.")

col1, col2 = st.columns([2, 1])
with col1:
    fig_normal = masco.plot_normal_distribution(returns, stock_ticker=selected_stock_ticker)
    st.pyplot(fig_normal)

with col2:
    st.subheader("Distribution Parameters")
    mu, sigma = norm.fit(stock_returns)
    st.metric("Mean (μ)", f"{mu:.4f}")
    st.metric("Standard Deviation (σ)", f"{sigma:.4f}")
    st.metric("Variance (σ²)", f"{sigma**2:.4f}")

# Excel tutorial for normal distribution
with st.expander("📘 How to Create Normal Distribution Graph in Excel"):
    st.markdown("""
    **Step 1: Calculate Statistics**
    - Calculate mean: `=AVERAGE(returns_range)`
    - Calculate standard deviation: `=STDEV.S(returns_range)`
    - Create bins for histogram: Create a column with bin ranges (e.g., -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15)
    
    **Step 2: Create Histogram**
    - Select your returns data
    - Go to Insert → Charts → Histogram (or use Data Analysis → Histogram)
    - If using Data Analysis:
      - Input Range: Select your returns data
      - Bin Range: Select your bin ranges
      - Check 'Chart Output'
    - Format the histogram: Right-click → Format Data Series → Adjust gap width
    
    **Step 3: Add Normal Distribution Curve**
    - Create a new column for normal distribution values
    - Use formula: `=NORM.DIST(x, mean, std_dev, FALSE)` where:
      - x = bin value
      - mean = calculated mean from Step 1
      - std_dev = calculated standard deviation from Step 1
      - FALSE = returns probability density (not cumulative)
    - Create a scatter plot with your bin values and normal distribution values
    - Add this as a line series to your histogram chart
    - Right-click chart → Select Data → Add → Select normal distribution data
    
    **Step 4: Format the Chart**
    - Add chart title: 'Stock Returns with Normal Distribution Fit'
    - Label axes: X-axis = 'Monthly Return', Y-axis = 'Density'
    - Add legend to distinguish histogram from normal curve
    - Format histogram bars: Set transparency (alpha) to ~60%
    - Format normal curve: Use a different color (e.g., red) with thicker line
    """)

st.divider()

# ===== SECTION 3: CDF GRAPH =====
st.header("📈 Cumulative Distribution Function (CDF)")
st.markdown(f"Comparison of cumulative distribution of monthly returns: {selected_stock_ticker} vs S&P 500")

col1, col2 = st.columns([2, 1])
with col1:
    fig_cdf = masco.plot_cdf(returns, stock_ticker=selected_stock_ticker)
    st.pyplot(fig_cdf)

with col2:
    st.subheader("Comparison Statistics")
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.markdown(f"**{selected_stock_ticker}**")
        st.write(f"Mean: {stock_returns.mean():.4f}")
        st.write(f"Std Dev: {stock_returns.std():.4f}")
        st.write(f"Min: {stock_returns.min():.4f}")
        st.write(f"Max: {stock_returns.max():.4f}")
    
    with col_stat2:
        st.markdown("**S&P 500**")
        st.write(f"Mean: {sp.mean():.4f}")
        st.write(f"Std Dev: {sp.std():.4f}")
        st.write(f"Min: {sp.min():.4f}")
        st.write(f"Max: {sp.max():.4f}")

# Excel tutorial for CDF
with st.expander("📘 How to Create CDF Graph in Excel"):
    st.markdown("""
    **Step 1: Prepare Data**
    - Sort your returns data in descending order: Select data → Data → Sort → Largest to Smallest
    - Create a column for cumulative probability
    - Use formula: `=1-ROW()/COUNT($A$2:$A$N)` where N is your last row
    - Or use: `=1-(ROW()-1)/(COUNT(A:A)-1)` if starting from row 2
    - This creates values from 0 to 1 representing cumulative probability
    
    **Step 2: Create Scatter Plot**
    - Select both columns: sorted returns and cumulative probability
    - Go to Insert → Charts → Scatter → Scatter with Markers
    - Excel will create a scatter plot with your data points
    
    **Step 3: Add Second Series (for Comparison)**
    - Right-click on the chart → Select Data
    - Click 'Add' to add a new series
    - Series name: 'S&P 500' (or your comparison stock)
    - X values: Select sorted S&P 500 returns
    - Y values: Select cumulative probability for S&P 500
    - Click OK to add the series
    
    **Step 4: Format the Chart**
    - Change marker styles:
      - Right-click first series → Format Data Series → Marker Options
      - Choose different marker (e.g., triangles ^ for Stock)
      - Right-click second series → Choose different marker (e.g., squares for S&P 500)
    - Change colors:
      - Format Data Series → Marker Fill → Choose colors (e.g., red for Stock, blue for S&P 500)
    - Add chart title: 'Distribution of Monthly Returns: Stock vs S&P 500'
    - Label axes:
      - X-axis: 'Return'
      - Y-axis: 'Cumulative Probability'
    - Add legend: Chart Tools → Add Chart Element → Legend
    
    **Step 5: Adjust Transparency**
    - Right-click each data series → Format Data Series
    - Go to Marker Fill → Transparency
    - Set transparency to ~30% (0.3) for better visibility
    - Adjust marker size if needed: Marker Options → Size
    
    **Step 6: Add Gridlines (Optional)**
    - Right-click chart → Add Chart Element → Gridlines
    - Choose Primary Major Horizontal and/or Vertical gridlines
    - This helps with reading values from the chart
    """)

# Footer
st.divider()
st.markdown("---")
st.markdown("**Data Source:** Yahoo Finance | **Analysis Period:** 2005-2025")

# Download section
st.divider()
st.subheader("📥 Download Results")

# Generate PDF report
with st.spinner("Preparing PDF download..."):
    pdf_buffer = masco.generate_pdf_report(
        returns, regressions, fig_normal, fig_cdf,
        start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'),
        stock_ticker=selected_stock_ticker, stock_name=selected_stock_name
    )
    pdf_bytes = pdf_buffer.getvalue()

# Main PDF download button (prominent)
col_main, col_side = st.columns([2, 1])

with col_main:
    # Create filename with stock ticker
    safe_stock_name = selected_stock_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    pdf_filename = f"{selected_stock_ticker}_{safe_stock_name}_report_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.pdf"
    
    st.download_button(
        label="📄 Download Full Report as PDF",
        data=pdf_bytes,
        file_name=pdf_filename,
        mime="application/pdf",
        help="Download complete report with Stargazer table, normal distribution graph, and CDF graph as PDF",
        use_container_width=True,
        type="primary"
    )
    st.markdown("**Includes:** Stargazer regression table, Normal distribution graph, and CDF graph")

with col_side:
    # Additional download options
    with st.expander("Other Downloads"):
        # Download returns data as CSV
        csv_returns = returns.to_csv()
        safe_stock_name = selected_stock_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        csv_filename = f"{selected_stock_ticker}_{safe_stock_name}_returns_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
        st.download_button(
            label="📊 Returns Data (CSV)",
            data=csv_returns,
            file_name=csv_filename,
            mime="text/csv",
            use_container_width=True
        )
        
        # Download regression summary as CSV
        reg_summary = []
        for i, reg in enumerate(regressions, 1):
            reg_summary.append({
                'Model': f'({i})',
                'R-squared': f"{reg.rsquared:.4f}",
                'Adj R-squared': f"{reg.rsquared_adj:.4f}",
                'F-statistic': f"{reg.fvalue:.4f}",
                'Observations': reg.nobs
            })
        reg_df = pd.DataFrame(reg_summary)
        csv_reg = reg_df.to_csv(index=False)
        reg_filename = f"{selected_stock_ticker}_{safe_stock_name}_regression_summary_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
        st.download_button(
            label="📈 Regression Summary (CSV)",
            data=csv_reg,
            file_name=reg_filename,
            mime="text/csv",
            use_container_width=True
        )

# Print button and print styles
print_css_js = """
<style>
    @media print {
        /* Page setup for 2 pages */
        @page {
            size: letter;
            margin: 0.75in;
        }
        
        /* Hide sidebar and UI elements */
        header[data-testid="stHeader"],
        [data-testid="stSidebar"],
        [data-testid="stToolbar"],
        .stDeployButton,
        #print-button-container {
            display: none !important;
            visibility: hidden !important;
        }
        
        /* Ensure main content is visible */
        .main .block-container {
            max-width: 100% !important;
            padding: 0 !important;
        }
        
        /* Make sure all content is visible */
        body, html, .stApp {
            visibility: visible !important;
            background: white !important;
        }
        
        /* Force page break after regression section */
        .print-page-break-1 {
            page-break-after: always !important;
            display: block !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Ensure content fits on pages */
        .element-container {
            page-break-inside: avoid;
            margin-bottom: 0.3em !important;
        }
        
        /* Table styling for print - ensure visibility */
        .stargazer-table,
        .stargazer-table * {
            color: #000000 !important;
            background-color: #ffffff !important;
            visibility: visible !important;
        }
        
        .stargazer-table table {
            width: 100% !important;
            font-size: 9pt !important;
            border-collapse: collapse !important;
        }
        
        /* Graph containers - ensure visibility */
        [data-testid="stImage"],
        [data-testid="stPyplot"],
        .stImage,
        img,
        canvas {
            page-break-inside: avoid !important;
            max-width: 100% !important;
            height: auto !important;
            visibility: visible !important;
            display: block !important;
        }
        
        /* Text content visibility */
        h1, h2, h3, h4, p, div, span, td, th {
            color: #000000 !important;
            visibility: visible !important;
        }
        
        /* Optimize spacing */
        .stMarkdown {
            margin-bottom: 0.3em !important;
        }
        
        h1, h2, h3 {
            page-break-after: avoid;
            margin-top: 0.5em !important;
            margin-bottom: 0.3em !important;
        }
        
        /* Reduce padding in columns */
        [data-testid="column"] {
            padding: 0.2em !important;
        }
        
        /* Hide expanders in print */
        [data-testid="stExpander"] {
            display: none !important;
        }
        
        /* Ensure metrics are visible */
        [data-testid="stMetric"] {
            visibility: visible !important;
        }
    }
    
    /* Print button styling */
    #print-button-container {
        text-align: center;
        margin: 30px 0;
        padding: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    
    #print-button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: 500;
        border-radius: 5px;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        transition: all 0.3s;
    }
    
    #print-button:hover {
        background-color: #1565a0;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    #print-button:active {
        background-color: #0d4d7a;
        transform: translateY(0);
    }
</style>

<div id="print-button-container">
    <button id="print-button" onclick="window.print()">
        📄 Download as PDF / Print
    </button>
    <p style="margin-top: 10px; color: #666; font-size: 14px;">
        Click to open print dialog - select "Save as PDF" to download (optimized for 2 pages)
    </p>
</div>

<script>
    window.addEventListener('load', function() {
        console.log('Print functionality ready');
    });
    
    // Ensure content is visible when printing
    window.addEventListener('beforeprint', function() {
        document.body.style.visibility = 'visible';
        var elements = document.querySelectorAll('.main, .block-container, .element-container');
        elements.forEach(function(el) {
            el.style.visibility = 'visible';
        });
    });
</script>
"""

html(print_css_js, height=100)

