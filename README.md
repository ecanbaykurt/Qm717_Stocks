# Stock Returns Analysis Application

A comprehensive web application for analyzing stock returns using regression analysis, normal distribution fitting, and cumulative distribution functions (CDF).

## Features

- 📊 **Interactive Regression Analysis**: Compare stock returns against market factors (S&P 500, Value-Weighted Market, 30-Year Treasury)
- 📈 **Multiple Stock Support**: Analyze any stock from a predefined list
- 📉 **Normal Distribution Visualization**: Histogram with fitted normal distribution curve
- 📈 **CDF Comparison**: Compare cumulative distributions between stocks and S&P 500
- 📄 **PDF Report Generation**: Download comprehensive reports with all analysis results
- 📘 **Excel Tutorials**: Step-by-step guides for replicating analysis in Excel

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd stock_returns
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

- `app.py` - Main Streamlit application
- `masco_2025.py` - Core analysis functions and PDF generation
- `stocks.py` - Stock ticker definitions
- `requirements.txt` - Python dependencies

## Features in Detail

### Regression Analysis
- Five regression models comparing stock returns to various market factors
- Statistical significance indicators (***, **, *)
- Standard errors and R-squared values

### Visualizations
- Normal distribution fit with histogram
- CDF comparison graphs
- Interactive date range selection

### PDF Reports
- Complete analysis reports
- Excel tutorials included
- Educational content explaining results

## Requirements

- Python 3.7+
- See `requirements.txt` for full list of dependencies

## License

[Add your license here]

