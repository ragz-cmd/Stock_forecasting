import yfinance as yf
import pandas as pd
from datetime import datetime
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
import psutil  # to check system load

def train_svr_model(stock_code):
    """
    Downloads stock data for the last year (1y) and trains an SVR model using GridSearchCV.
    Uses a pipeline to scale input features (X) and a TransformedTargetRegressor to scale the target (y).
    Checks the system's CPU load to adjust the parameter grid size.
    Returns the best model, MSE, MAE, and the original DataFrame.
    """
    # Download data for the last year with daily intervals
    df = yf.download(stock_code, period="1y", interval="1d")
    if df.empty:
        raise ValueError("No data available for the given stock code.")
    
    # Use only Date and Close columns, then convert Date to a numeric ordinal
    df = df.reset_index()[['Date', 'Close']]
    df['Date_numeric'] = df['Date'].apply(lambda x: x.toordinal())
    X = df['Date_numeric'].values.reshape(-1, 1)
    y = df['Close'].values.ravel()  # Ensure y is 1D

    # Split data into training and testing sets (90% train, 10% test) without shuffling (for time-series)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

    # Create a pipeline for the SVR model that scales X
    svr = SVR(kernel='rbf')
    pipeline = make_pipeline(StandardScaler(), svr)
    # Use TransformedTargetRegressor to automatically scale the target variable (y)
    regressor = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())

    # Check CPU load to adjust the grid search parameter grid size
    cpu_load = psutil.cpu_percent(interval=1)
    print(f"CPU Load: {cpu_load}%")
    if cpu_load > 80:
        param_grid = {
            'regressor__svr__C': [1, 10],
            'regressor__svr__epsilon': [0.1, 0.5],
            'regressor__svr__gamma': ['scale', 0.1]
        }
    else:
        param_grid = {
            'regressor__svr__C': [0.1, 1, 10, 100],
            'regressor__svr__epsilon': [0.01, 0.1, 0.5, 1],
            'regressor__svr__gamma': ['scale', 'auto', 0.01, 0.1, 1]
        }

    grid = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Evaluate model performance on the test set
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return best_model, mse, mae, df

def create_forecast(model, df, forecast_days):
    """
    Generates a forecast using the trained model.
    Creates future dates starting from the last date in df and predicts the stock price.
    """
    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_days + 1)]
    future_dates_numeric = [[d.toordinal()] for d in future_dates]
    predictions = model.predict(future_dates_numeric)
    
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted": predictions
    })
    return forecast_df
