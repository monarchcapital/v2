# utils.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
import traceback
from sklearn.decomposition import PCA

def display_log(message: str, level: str = "info"):
    """
    Displays a log message in the Streamlit app's log expander.
    """
    if 'log_history' not in st.session_state:
        st.session_state['log_history'] = []
    
    st.session_state['log_history'].append({'message': message, 'level': level})
    if level == "error":
        print(f"ERROR: {message}")
    elif level == "warning":
        print(f"WARNING: {message}")
    else:
        print(f"INFO: {message}")


def preprocess_data(df: pd.DataFrame, original_close_data: np.ndarray = None, apply_differencing: bool = False):
    """
    Preprocesses the data by handling NaNs, scaling features, and preparing for sequence creation.
    """
    display_log("🔄 Preprocessing Data...", "info")
    df_copy = df.copy()

    if df_copy.empty:
        display_log("❗ Input DataFrame is empty for preprocessing.", "warning")
        return pd.DataFrame(), None, -1, pd.Series(), np.nan

    if 'Close' not in df_copy.columns:
        display_log("❌ Critical Error: 'Close' column not found for preprocessing.", "error")
        return pd.DataFrame(), None, -1, pd.Series(), np.nan
    
    initial_close_value = df_copy['Close'].iloc[0] if apply_differencing else np.nan
    last_actual_values = df_copy.iloc[-1].copy()

    try:
        df_copy.dropna(inplace=True)
        df_copy.fillna(method='ffill', inplace=True)
        df_copy.fillna(method='bfill', inplace=True)

        if df_copy.isnull().sum().sum() > 0:
            df_copy.dropna(inplace=True)
        
        if apply_differencing:
            if len(df_copy) < 2:
                display_log("❗ Not enough data to apply differencing. Skipping.", "warning")
            else:
                df_copy = df_copy.diff().dropna()
                display_log(f"✅ Differencing applied. New shape: {df_copy.shape}", "info")

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(df_copy.values)
        
        scaled_values = scaler.transform(df_copy.values)
        scaled_df = pd.DataFrame(scaled_values, columns=df_copy.columns, index=df_copy.index)

        if 'Close' in scaled_df.columns:
            scaled_df.rename(columns={'Close': 'Close_scaled'}, inplace=True)
            close_col_index = scaled_df.columns.get_loc('Close_scaled')
        else:
            close_col_index = -1

        display_log(f"✅ Data preprocessing complete. Scaled shape: {scaled_df.shape}", "info")
        return scaled_df, scaler, close_col_index, last_actual_values, initial_close_value
    except Exception as e:
        display_log(f"❌ Error during data preprocessing: {e}", "error")
        st.exception(e)
        return pd.DataFrame(), None, -1, pd.Series(), np.nan


def apply_pca(df: pd.DataFrame, n_components: int = 0):
    """
    Applies Principal Component Analysis (PCA) to the DataFrame.
    Returns the PCA-transformed DataFrame and the fitted pca object.
    """
    display_log("📉 Applying PCA...", "info")
    if df.empty:
        display_log("❗ Input DataFrame is empty for PCA. Skipping.", "warning")
        return pd.DataFrame(), None

    try:
        # If n_components is 0, it's 'auto' mode (95% variance)
        # Otherwise, it's the user-defined number of components
        pca = PCA(n_components=n_components if n_components > 0 else 0.95)
        
        if n_components > 0 and n_components > df.shape[1]:
            display_log(f"❗ n_components ({n_components}) > features ({df.shape[1]}). Adjusting.", "warning")
            pca = PCA(n_components=df.shape[1])

        pca_features = pca.fit_transform(df)
        
        pca_df = pd.DataFrame(data=pca_features, index=df.index,
                              columns=[f'PC_{i+1}' for i in range(pca.n_components_)])
        
        display_log(f"✅ PCA applied. Original features: {df.shape[1]}, Components selected: {pca.n_components_}", "info")
        return pca_df, pca
    except Exception as e:
        display_log(f"❌ Error during PCA: {e}", "error")
        st.exception(e)
        return pd.DataFrame(), None

def create_sequences(data: np.ndarray, target_column_index: int, time_steps: int, prediction_horizon: int):
    """
    Creates sequences (X) and corresponding target vectors (y) for multi-step forecasting.
    'y' will be a vector of the next 'prediction_horizon' steps.
    """
    display_log(f"🔄 Creating sequences with {time_steps} time steps for a {prediction_horizon}-day forecast.", "info")
    X, y = [], []

    if len(data) < time_steps + prediction_horizon:
        display_log(f"❗ Not enough data to create sequences. Data length: {len(data)}, Required: {time_steps + prediction_horizon}.", "warning")
        return np.array([]), np.array([])

    try:
        for i in range(len(data) - time_steps - prediction_horizon + 1):
            X.append(data[i:(i + time_steps), :])
            y.append(data[i + time_steps : i + time_steps + prediction_horizon, target_column_index])

        X = np.array(X)
        y = np.array(y) # y is now 2D: (samples, prediction_horizon)

        display_log(f"✅ Sequences created. X shape: {X.shape}, y shape: {y.shape}", "info")
        return X, y
    except Exception as e:
        display_log(f"❌ Error creating sequences: {e}. Traceback: {traceback.format_exc()}", "error")
        st.exception(e)
        return np.array([]), np.array([])


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray):
    """
    Calculates RMSE, MAE, and R2 score.
    """
    display_log("🧮 Calculating Metrics...", "info")
    try:
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        display_log(f"✅ Metrics: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}", "info")
        return rmse, mae, r2
    except Exception as e:
        display_log(f"❌ Error calculating metrics: {e}", "error")
        return np.nan, np.nan, np.nan


def plot_predictions(actual_prices: pd.Series, predicted_prices: pd.Series, ticker: str, 
                     future_predictions: pd.Series = None, confidence_upper: pd.Series = None, 
                     confidence_lower: pd.Series = None, plot_title_suffix: str = ""):
    """
    Plots historical, predicted, and future prices using Plotly.
    """
    display_log(f"📈 Plotting predictions for {ticker}...", "info")
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=actual_prices.index, y=actual_prices, mode='lines', name='Actual Price (Test Set)',
                             line=dict(color='blue')))

    fig.add_trace(go.Scatter(x=predicted_prices.index, y=predicted_prices, mode='lines', name='Predicted Price (Test Set)',
                             line=dict(color='red', dash='dash')))

    if future_predictions is not None and not future_predictions.empty:
        fig.add_trace(go.Scatter(x=future_predictions.index, y=future_predictions, mode='lines', name='Future Prediction',
                                 line=dict(color='green', dash='dot')))
        
        if confidence_upper is not None and confidence_lower is not None:
            fig.add_trace(go.Scatter(
                x=future_predictions.index.tolist() + future_predictions.index.tolist()[::-1],
                y=confidence_upper.tolist() + confidence_lower.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))

    fig.update_layout(
        title=f'{ticker} Stock Price Prediction (Test and Future) {plot_title_suffix}',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    display_log(f"✅ Plot generated for {ticker}.", "info")
    return fig