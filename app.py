import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from datetime import datetime as dt
import yfinance as yf
import pandas as pd
import plotly.express as px

# Create a Dash instance and expose the Flask server
app = dash.Dash(__name__)
server = app.server

# Build the application layout
app.layout = html.Div([
    # Parent container: wraps the navigation and content sections
    html.Div([
        # Navigation section: inputs and buttons
        html.Div(
            [
                html.P("Welcome to the Stock Dash App!", className="start"),
                html.Div(
                    [
                        # Stock code input
                        dcc.Input(
                            id="stock-code",
                            type="text",
                            placeholder="Enter Stock Code",
                            className="input-field"
                        )
                    ],
                    className="input-stock"
                ),
                html.Div(
                    [
                        # Date range picker input
                        dcc.DatePickerRange(
                            id="date-range",
                            start_date=dt(2020, 1, 1),
                            end_date=dt.now().date(),
                            display_format="YYYY-MM-DD"
                        )
                    ],
                    className="input-date"
                ),
                html.Div(
                    [
                        # Buttons and forecast input (forecast functionality to be implemented later)
                        html.Button("Stock Price", id="stock-price-btn", className="btn"),
                        html.Button("Indicators", id="indicators-btn", className="btn"),
                        dcc.Input(
                            id="forecast-days",
                            type="number",
                            placeholder="Days of Forecast",
                            className="input-number"
                        ),
                        html.Button("Forecast", id="forecast-btn", className="btn")
                    ],
                    className="input-actions"
                )
            ],
            className="nav"
        ),
        # Content section: for company info and graphs
        html.Div(
            [
                html.Div(
                    [
                        # Company logo and name; defaults to a placeholder image
                        html.Img(src="assets/default_logo.png", id="company-logo", style={"height": "50px"}),
                        html.H1("Company Name", id="company-name")
                    ],
                    className="header"
                ),
                # Company description to be updated via callback
                html.Div(id="description", className="description_ticker"),
                # Div for stock price graph
                html.Div(id="graphs-content"),
                # Div for indicator graph (EMA)
                html.Div(id="main-content"),
                # Div for forecast graph (to be implemented later)
                html.Div(id="forecast-content")
            ],
            className="content"
        )
    ], className="container")
])

# Helper function: Flatten multi-index columns, if present
def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns.values]
    return df

# Callback 1: Update company information (description, logo, company name)
@app.callback(
    [Output("description", "children"),
     Output("company-logo", "src"),
     Output("company-name", "children")],
    [Input("stock-price-btn", "n_clicks")],
    [State("stock-code", "value")]
)
def update_company_info(n_clicks, stock_code):
    if not n_clicks or not stock_code:
        raise PreventUpdate

    try:
        ticker = yf.Ticker(stock_code)
        info = ticker.info
    except Exception as e:
        return f"Error fetching info: {e}", "assets/default_logo.png", "Unknown Company"

    description = info.get("longBusinessSummary", "No description available.")
    logo = info.get("logo_url", "")
    default_logo = "assets/default_logo.png"
    # Use default logo if logo URL is missing or invalid
    if not logo or not logo.startswith("http"):
        logo = default_logo
    name = info.get("shortName", "Unknown Company")
    return description, logo, name

# Callback 2: Update stock price graph
@app.callback(
    Output("graphs-content", "children"),
    [Input("stock-price-btn", "n_clicks")],
    [State("stock-code", "value"),
     State("date-range", "start_date"),
     State("date-range", "end_date")]
)
def update_stock_price(n_clicks, stock_code, start_date, end_date):
    if not n_clicks or not stock_code:
        raise PreventUpdate

    try:
        start_date_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end_date_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        # Set auto_adjust explicitly to False for consistency
        df = yf.download(stock_code, start=start_date_str, end=end_date_str, auto_adjust=False)
    except Exception as e:
        return html.Div(f"Error downloading stock data: {e}")

    if df.empty:
        return html.Div("No data available for this stock and date range.")

    df.reset_index(inplace=True)
    df = flatten_columns(df)
    fig = get_stock_price_fig(df)
    return dcc.Graph(figure=fig)

# Callback 3: Update indicator graph (Exponential Moving Average)
@app.callback(
    Output("main-content", "children"),
    [Input("indicators-btn", "n_clicks")],
    [State("stock-code", "value"),
     State("date-range", "start_date"),
     State("date-range", "end_date")]
)
def update_indicator(n_clicks, stock_code, start_date, end_date):
    if not n_clicks or not stock_code:
        raise PreventUpdate

    try:
        start_date_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end_date_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        df = yf.download(stock_code, start=start_date_str, end=end_date_str, auto_adjust=False)
    except Exception as e:
        return html.Div(f"Error downloading stock data: {e}")

    if df.empty:
        return html.Div("No data available for this stock and date range.")

    df.reset_index(inplace=True)
    df = flatten_columns(df)
    fig = get_indicator_fig(df)
    return dcc.Graph(figure=fig)

# Helper function: Generate stock price graph using Plotly Express
def get_stock_price_fig(df):
    # Create a line chart of Open and Close prices versus Date
    fig = px.line(df,
                  x="Date",
                  y=["Open", "Close"],
                  title="Opening and Closing Price vs Date")
    return fig

# Helper function: Generate indicator graph (20-day Exponential Moving Average)
def get_indicator_fig(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.line(df,
                  x="Date",
                  y="EWA_20",
                  title="Exponential Moving Average vs Date")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
