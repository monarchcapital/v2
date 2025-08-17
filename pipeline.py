# pipeline.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# --- Import Core Functions from other modules ---
from data import fetch_stock_data, add_technical_indicators, add_macro_indicators, add_fundamental_indicators
from utils import preprocess_data, apply_pca, create_sequences, calculate_metrics, display_log
from model import build_lstm_model, build_transformer_model, build_hybrid_model, train_model, predict_prices, run_hyperparameter_tuning, KERAS_TUNER_AVAILABLE

# --- Centralized Prediction Pipeline ---

def run_prediction_pipeline(
    ticker_symbol, start_date, end_date, selected_indicators,
    selected_macro_indicators, selected_fundamental_indicators,
    apply_differencing, enable_pca, n_components_manual,
    training_window_size, test_set_split_ratio, prediction_horizon,
    model_type, hp_mode, force_retune, epochs, num_trials,
    executions_per_trial, manual_params, batch_size, learning_rate,
    num_ensemble_models, best_models_dir="best_models"
):
    """
    Executes the entire end-to-end prediction workflow for a given stock ticker.
    This function centralizes logic from ML_Prediction.py and Watchlist.py.
    """
    try:
        # --- 1. Fetch and Prepare Data ---
        display_log(f"--- Starting Pipeline for {ticker_symbol} ---", "info")
        data = fetch_stock_data(ticker_symbol, start_date, end_date)
        if data.empty:
            return {'error': f"Data fetching failed for {ticker_symbol}."}

        if selected_indicators: data = add_technical_indicators(data, selected_indicators)
        if selected_macro_indicators: data = add_macro_indicators(data, selected_macro_indicators)
        if selected_fundamental_indicators: data = add_fundamental_indicators(data, ticker_symbol, selected_fundamental_indicators)
        if data.empty:
            return {'error': f"Data became empty after adding features for {ticker_symbol}."}

        # --- 2. Preprocess Data ---
        processed_data, scaler, close_col_index, last_actual_values, was_differenced = preprocess_data(data.copy(), apply_differencing=apply_differencing)
        if processed_data.empty or scaler is None:
            return {'error': f"Data preprocessing failed for {ticker_symbol}."}

        # --- 3. PCA (Optional) ---
        data_for_seq = processed_data
        pca_object = None
        original_feature_names = None
        if enable_pca and 'Close_scaled' in processed_data.columns:
            features_for_pca = processed_data.drop(columns=['Close_scaled'])
            if not features_for_pca.empty:
                pca_result, pca_obj = apply_pca(features_for_pca, n_components=n_components_manual)
                if not pca_result.empty and pca_obj is not None:
                    pca_object = pca_obj
                    original_feature_names = features_for_pca.columns.tolist()
                    pca_result.reset_index(drop=True, inplace=True)
                    close_scaled_series = processed_data['Close_scaled'].reset_index(drop=True)
                    data_for_seq = pd.concat([pca_result, close_scaled_series], axis=1)
                    display_log(f"✅ PCA applied. New data shape for sequencing: {data_for_seq.shape}", "info")

        # --- 4. Create Sequences ---
        if 'Close_scaled' not in data_for_seq.columns:
            return {'error': "Target column 'Close_scaled' not found after PCA."}
        target_idx = data_for_seq.columns.get_loc('Close_scaled')
        X, y = create_sequences(data_for_seq.values, target_idx, training_window_size, prediction_horizon)
        if X.size == 0:
            return {'error': f"Failed to create sequences for {ticker_symbol}."}
        test_size = int(len(X) * test_set_split_ratio)
        X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]

        # --- 5. Hyperparameter Tuning or Manual Build ---
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_dim = prediction_horizon
        best_hps = None
        if hp_mode == "Automatic (Tuner)" and KERAS_TUNER_AVAILABLE:
            display_log("🚀 Starting Hyperparameter Tuning...", "info")
            _, best_hps, tuner_params, _ = run_hyperparameter_tuning(
                model_type, input_shape, output_dim, X_train, y_train, num_trials, executions_per_trial, best_models_dir, force_retune, epochs)
            if tuner_params:
                learning_rate = tuner_params.get('learning_rate', learning_rate)
                batch_size = tuner_params.get('batch_size', batch_size)
                display_log(f"✅ Tuner finished. Using LR: {learning_rate}, Batch Size: {batch_size}", "info")
        else:
            display_log("🛠️ Using Manual Hyperparameters.", "info")

        # --- 6. Build, Train, Predict (Ensemble Loop) ---
        all_future_preds, all_test_preds_scaled, trained_models_list = [], [], []
        model_builder = {'LSTM': build_lstm_model, 'Transformer': build_transformer_model, 'Hybrid (LSTM + Transformer)': build_hybrid_model}
        for i in range(num_ensemble_models):
            display_log(f"Training model {i+1}/{num_ensemble_models}...", "info")
            model = model_builder[model_type](input_shape, output_dim, hp=best_hps, manual_params=manual_params, learning_rate=learning_rate)
            if model:
                train_model(model, X_train, y_train, epochs, batch_size, learning_rate, X_val=X_test, y_val=y_test)
                future_preds = predict_prices(model, data_for_seq, scaler, close_col_index, training_window_size, prediction_horizon, last_actual_values, was_differenced=was_differenced)
                if future_preds.size > 0: all_future_preds.append(future_preds)
                if X_test.size > 0: all_test_preds_scaled.append(model.predict(X_test, verbose=0))
                trained_models_list.append(model)

        # --- 7. Process and Aggregate Results ---
        results = {
            'data': data, 'y_test_len': len(y_test), 'trained_models': trained_models_list,
            'future_predicted_prices': None, 'test_actual_prices': None,
            'test_predicted_prices': None, 'model_metrics': {}, 'pca_object': pca_object,
            'original_feature_names': original_feature_names
        }

        if all_future_preds:
            avg_future_preds = np.mean(all_future_preds, axis=0)
            last_date = data.index[-1]
            future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=prediction_horizon)
            results['future_predicted_prices'] = pd.Series(avg_future_preds, index=future_dates)

        if all_test_preds_scaled and y_test.size > 0:
            avg_test_preds_scaled = np.mean(all_test_preds_scaled, axis=0)[:, 0]
            y_test_first_step_scaled = y_test[:, 0]
            actual_dummy = np.zeros((len(y_test_first_step_scaled), scaler.n_features_in_))
            predicted_dummy = np.zeros((len(avg_test_preds_scaled), scaler.n_features_in_))
            actual_dummy[:, close_col_index] = y_test_first_step_scaled
            predicted_dummy[:, close_col_index] = avg_test_preds_scaled
            actual_prices_unscaled = scaler.inverse_transform(actual_dummy)[:, close_col_index]
            predicted_prices_unscaled = scaler.inverse_transform(predicted_dummy)[:, close_col_index]
            
            if was_differenced:
                # Find the correct starting point for inverse differencing
                last_actual_price = data['Close'].iloc[-len(y_test)-1]
                actual_prices_inv = last_actual_price + np.cumsum(actual_prices_unscaled)
                predicted_prices_inv = last_actual_price + np.cumsum(predicted_prices_unscaled)
            else:
                actual_prices_inv, predicted_prices_inv = actual_prices_unscaled, predicted_prices_unscaled

            test_dates = data.index[-len(y_test):]
            results['test_actual_prices'] = pd.Series(actual_prices_inv, index=test_dates)
            results['test_predicted_prices'] = pd.Series(predicted_prices_inv, index=test_dates)
            rmse, mae, r2 = calculate_metrics(actual_prices_inv, predicted_prices_inv)
            results['model_metrics'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

        # --- 8. Fetch News ---
        try:
            GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel(model_name='gemini-1.5-flash')
            prompt = f"As an expert financial market analyst, generate a concise market summary based on the absolute latest news for the stock '{ticker_symbol}'. Structure it with a Main Headline, 2-3 Key Takeaways in bullets, and a Brief Analysis paragraph. Use Markdown."
            response = model.generate_content(prompt)
            results['news_summary'] = response.text
        except (KeyError, FileNotFoundError):
            results['news_summary'] = "Google AI API Key not found. News analysis disabled."
        except google_exceptions.GoogleAPICallError as e:
            results['news_summary'] = f"Could not generate summary due to a Google API error: {e}"
        except Exception as e:
            results['news_summary'] = f"An unexpected error occurred while generating the news summary: {e}"

        display_log(f"--- Pipeline for {ticker_symbol} Completed Successfully ---", "info")
        return results

    except Exception as e:
        display_log(f"--- Pipeline for {ticker_symbol} Failed: {e} ---", "error")
        return {'error': str(e)}
