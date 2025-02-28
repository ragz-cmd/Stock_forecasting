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

# Updated flatten_columns function to convert multi-index columns to single-level strings
def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        new_columns = []
        for col in df.columns:
            # If second element is empty, use only the first element;
            # otherwise, use the first element (since we assume one ticker)
            new_columns.append(col[0])
        df.columns = new_columns
    return df

# Build the application layout
app.layout = html.Div([
    # Parent container: wraps navigation and content sections
    html.Div([
        # Navigation section: inputs and buttons
        html.Div(
            [
                html.P("Welcome to the Stock Dash App!", className="start"),
                html.Div(
                    [
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
                        html.Img(src="/assets/default_logo.png", id="company-logo", style={"height": "50px"}),
                        html.H1("Company Name", id="company-name")
                    ],
                    className="header"
                ),
                html.Div(id="description", className="description_ticker"),
                html.Div(id="graphs-content"),
                html.Div(id="main-content"),
                html.Div(id="forecast-content")
            ],
            className="content"
        )
    ], className="container")
])

# Callback 1: Update company information
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
        return f"Error fetching info: {e}", "/assets/default_logo.png", "Unknown Company"

    description = info.get("longBusinessSummary", "No description available.")
    logo = info.get("logo_url", "")
    if not logo or not logo.startswith("http"):
        website = info.get("website", "")
        if website:
            domain = website.replace("https://", "").replace("http://", "").split("/")[0]
            logo = f"https://logo.clearbit.com/{domain}"
        else:
            logo = "/assets/default_logo.png"
    name = info.get("shortName", "Unknown Company")
    print("Logo URL:", logo)  # Debug print
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
        df = yf.download(stock_code, start=start_date_str, end=end_date_str, auto_adjust=False)
    except Exception as e:
        return html.Div(f"Error downloading stock data: {e}")

    if df.empty:
        return html.Div("No data available for this stock and date range.")
    
    df.reset_index(inplace=True)
    df = flatten_columns(df)
    fig = px.line(df, x="Date", y=["Open", "Close"], title="Opening and Closing Price vs Date")
    return dcc.Graph(figure=fig)

# Callback 3: Update indicator graph (EMA)
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
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.line(df, x="Date", y="EWA_20", title="Exponential Moving Average vs Date")
    return dcc.Graph(figure=fig)

# Callback 4: Forecast using the SVR model (imported from model.py)
@app.callback(
    Output("forecast-content", "children"),
    [Input("forecast-btn", "n_clicks")],
    [State("stock-code", "value"),
     State("forecast-days", "value")]
)
def forecast_stock(n_clicks, stock_code, forecast_days):
    if not n_clicks or not stock_code or not forecast_days:
        raise PreventUpdate
    try:
        from model import train_svr_model, create_forecast
        model, mse, mae, df = train_svr_model(stock_code)
    except Exception as e:
        return html.Div(f"Error training model: {e}")
    
    forecast_df = create_forecast(model, df, int(forecast_days))
    fig = px.line(forecast_df, x="Date", y="Predicted", title="Forecasted Stock Prices")
    metrics_text = f"Model Performance - MSE: {mse:.2f}, MAE: {mae:.2f}"
    return html.Div([html.P(metrics_text), dcc.Graph(figure=fig)])

if __name__ == '__main__':
    app.run_server(debug=True)
