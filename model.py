# model.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer, Input, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
import traceback
import os
import json
from sklearn.preprocessing import MinMaxScaler

from utils import display_log, calculate_metrics

try:
    import keras_tuner as kt
    KERAS_TUNER_AVAILABLE = True
except ImportError:
    KERAS_TUNER_AVAILABLE = False
    display_log("The 'keras_tuner' library is not installed. Hyperparameter tuning will be skipped.", "error")

# --- Custom Transformer Layer (No changes needed here) ---
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        # Store parameters for get_config
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate


    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "embed_dim": self.embed_dim, "num_heads": self.num_heads,
            "ff_dim": self.ff_dim, "rate": self.rate,
        })
        return config

# --- Model Building Functions (Implemented) ---

def build_lstm_model(input_shape: tuple, output_dim: int, hp=None, manual_params: dict = None, learning_rate: float = 0.001):
    """Builds and compiles an LSTM model."""
    display_log("🏗️ Building LSTM Model...", "info")
    model = Sequential()
    
    params = manual_params or {}
    num_layers = params.get('num_lstm_layers', 2)
    
    for i in range(num_layers):
        units = params.get(f'lstm_units_{i+1}', 100)
        dropout = params.get(f'dropout_{i+1}', 0.2)
        return_sequences = (i < num_layers - 1)
        
        if i == 0:
            model.add(LSTM(units, return_sequences=return_sequences, input_shape=input_shape))
        else:
            model.add(LSTM(units, return_sequences=return_sequences))
        model.add(Dropout(dropout))
        
    model.add(Dense(output_dim))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    display_log("✅ LSTM Model Built.", "info")
    return model

def build_transformer_model(input_shape: tuple, output_dim: int, hp=None, manual_params: dict = None, learning_rate: float = 0.001):
    """Builds and compiles a Transformer-based model."""
    display_log("🏗️ Building Transformer Model...", "info")
    params = manual_params or {}
    num_blocks = params.get('num_transformer_blocks', 2)
    num_heads = params.get('num_heads', 4)
    ff_dim = params.get('ff_dim', 32)
    embed_dim = input_shape[1] # Number of features

    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    
    x = Flatten()(x)
    x = Dropout(0.1)(x)
    x = Dense(20, activation="relu")(x)
    outputs = Dense(output_dim)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    display_log("✅ Transformer Model Built.", "info")
    return model

def build_hybrid_model(input_shape: tuple, output_dim: int, hp=None, manual_params: dict = None, learning_rate: float = 0.001):
    """Builds and compiles a Hybrid LSTM-Transformer model."""
    display_log("🏗️ Building Hybrid (LSTM+Transformer) Model...", "info")
    params = manual_params or {}
    lstm_units = params.get('lstm_units', 64)
    lstm_dropout = params.get('lstm_dropout', 0.2)
    num_blocks = params.get('num_transformer_blocks', 1)
    num_heads = params.get('num_heads', 2)
    ff_dim = input_shape[1] # Match embed_dim

    inputs = Input(shape=input_shape)
    # LSTM part
    x = LSTM(lstm_units, return_sequences=True, dropout=lstm_dropout)(inputs)
    # Transformer part
    for _ in range(num_blocks):
        x = TransformerBlock(embed_dim=lstm_units, num_heads=num_heads, ff_dim=ff_dim)(x)
    
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(output_dim)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    display_log("✅ Hybrid Model Built.", "info")
    return model

# --- Training and Prediction (Implemented) ---

def train_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                epochs: int, batch_size: int, learning_rate: float,
                X_val: np.ndarray = None, y_val: np.ndarray = None):
    """Trains the Keras model with early stopping and learning rate reduction."""
    display_log(f"💪 Starting model training for {epochs} epochs...", "info")
    
    early_stopping = EarlyStopping(monitor='val_loss' if X_val is not None else 'loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss' if X_val is not None else 'loss', factor=0.2, patience=5, min_lr=1e-6)
    
    validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=[early_stopping, reduce_lr],
        verbose=1 
    )
    display_log("✅ Model training complete.", "info")
    return history

def predict_prices(model: tf.keras.Model, processed_data: pd.DataFrame, scaler: MinMaxScaler,
                   close_col_index: int, time_steps: int, prediction_horizon: int,
                   last_actual_values_before_diff: pd.Series = None,
                   was_differenced: bool = False):
    """Makes future price predictions using the trained model."""
    display_log("🔮 Generating future predictions...", "info")
    
    try:
        # Get the last 'time_steps' sequence from the data
        last_sequence = processed_data.values[-time_steps:]
        input_data = np.expand_dims(last_sequence, axis=0)

        # Predict the future 'prediction_horizon' steps
        predicted_scaled = model.predict(input_data)[0]

        # Inverse transform the predictions
        dummy_array = np.zeros((len(predicted_scaled), scaler.n_features_in_))
        dummy_array[:, close_col_index] = predicted_scaled
        predicted_unscaled = scaler.inverse_transform(dummy_array)[:, close_col_index]

        # FIX: Corrected comment to be more general
        # Inverse differencing if it was applied
        if was_differenced:
            last_actual_price = last_actual_values_before_diff['Close']
            predicted_prices = last_actual_price + np.cumsum(predicted_unscaled)
            display_log("✅ Inverse differencing applied to predictions.", "info")
        else:
            predicted_prices = predicted_unscaled
        
        display_log("✅ Future predictions generated successfully.", "info")
        return predicted_prices
    except Exception as e:
        display_log(f"❌ Error during prediction: {e}", "error")
        st.exception(e)
        return np.array([])


# --- Hyperparameter Tuning (Implemented) ---

if KERAS_TUNER_AVAILABLE:
    def build_model_for_tuning(hp, model_type, input_shape, output_dim):
        """Builds a model with hyperparameters for Keras Tuner."""
        # Define Hyperparameters
        learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
        
        manual_params = {}
        if model_type == 'LSTM':
            manual_params['num_lstm_layers'] = hp.Int('num_lstm_layers', 1, 3)
            for i in range(manual_params['num_lstm_layers']):
                manual_params[f'lstm_units_{i+1}'] = hp.Int(f'lstm_units_{i+1}', 32, 256, step=32)
                manual_params[f'dropout_{i+1}'] = hp.Float(f'dropout_{i+1}', 0.1, 0.5, step=0.1)
            model = build_lstm_model(input_shape, output_dim, manual_params=manual_params, learning_rate=learning_rate)
        elif model_type == 'Transformer':
            manual_params['num_transformer_blocks'] = hp.Int('num_transformer_blocks', 1, 3)
            manual_params['num_heads'] = hp.Int('num_heads', 2, 8, step=2)
            manual_params['ff_dim'] = hp.Int('ff_dim', 32, 128, step=32)
            model = build_transformer_model(input_shape, output_dim, manual_params=manual_params, learning_rate=learning_rate)
        else: # Hybrid
            manual_params['lstm_units'] = hp.Int('lstm_units', 32, 128, step=32)
            manual_params['lstm_dropout'] = hp.Float('lstm_dropout', 0.1, 0.5, step=0.1)
            manual_params['num_transformer_blocks'] = hp.Int('num_transformer_blocks', 1, 2)
            manual_params['num_heads'] = hp.Int('num_heads', 2, 4, step=2)
            model = build_hybrid_model(input_shape, output_dim, manual_params=manual_params, learning_rate=learning_rate)
            
        return model

    def run_hyperparameter_tuning(model_type: str, input_shape: tuple, output_dim: int,
                                  X_train: np.ndarray, y_train: np.ndarray,
                                  num_trials: int, executions_per_trial: int,
                                  best_models_dir: str, force_retune: bool = False,
                                  epochs: int = 50):
        project_name = f'tuner_{model_type}'
        tuner_dir = os.path.join(best_models_dir, project_name)

        if not force_retune and os.path.exists(tuner_dir):
            display_log("✅ Found existing tuner results. Loading best model.", "info")
            return load_best_tuner_model(model_type, input_shape, output_dim, best_models_dir)

        display_log(f"튜닝을 시작합니다: {model_type}...", "info")
        
        tuner = kt.RandomSearch(
            hypermodel=lambda hp: build_model_for_tuning(hp, model_type, input_shape, output_dim),
            objective='val_loss',
            max_trials=num_trials,
            executions_per_trial=executions_per_trial,
            directory=best_models_dir,
            project_name=project_name,
            overwrite=True
        )
        
        tuner.search(X_train, y_train, epochs=epochs, validation_split=0.2,
                     callbacks=[EarlyStopping('val_loss', patience=5)])
        
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.get_best_models(num_models=1)[0]
        
        learning_rate = best_hps.get('learning_rate')
        batch_size = best_hps.get('batch_size')

        display_log("✅ Hyperparameter tuning complete.", "info")
        return best_model, best_hps, {'learning_rate': learning_rate, 'batch_size': batch_size}, tuner.results_summary()

    def load_best_tuner_model(model_type: str, input_shape: tuple, output_dim: int, best_models_dir: str):
        """Loads the best model from a completed Keras Tuner run."""
        project_name = f'tuner_{model_type}'
        tuner_dir = os.path.join(best_models_dir, project_name)
        
        try:
            tuner = kt.RandomSearch(
                hypermodel=lambda hp: build_model_for_tuning(hp, model_type, input_shape, output_dim),
                objective='val_loss',
                directory=best_models_dir,
                project_name=project_name
            )
            best_model = tuner.get_best_models(num_models=1)[0]
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            learning_rate = best_hps.get('learning_rate')
            batch_size = best_hps.get('batch_size')
            
            display_log("✅ Best model and HPs loaded from tuner directory.", "info")
            return best_model, best_hps, {'learning_rate': learning_rate, 'batch_size': batch_size}
        except Exception as e:
            display_log(f"❌ Could not load from tuner directory: {e}. A retune might be needed.", "error")
            return None, None, {}
else:
    def run_hyperparameter_tuning(*args, **kwargs): return None, None, {}, {}
    def load_best_tuner_model(*args, **kwargs): return None, None, {}