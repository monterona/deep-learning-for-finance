# Copilot Instructions: Deep Learning for Finance

This repository implements deep learning models for financial prediction, organized by chapters covering progressively advanced techniques from statistical analysis to LSTM-based forecasting.

## Project Structure & Architecture

- **Chapter 1-3**: Data foundations - VIX analysis, statistical measures (central tendency, variability, stationarity)
- **Chapter 7**: Classical ML regressors (Linear, SVM, XGBoost, etc.)
- **Chapter 8**: Neural networks for regression (MLP, RNN, LSTM, CNN)
- **Chapter 9**: Advanced forecasting - Fractional differentiation, multi-period forecasting (MPF), continuous retraining
- **Chapter 10-11**: Advanced architectures (DDQN, LSTM with technical indicators, volatility models)
- **master_function.py**: Central hub containing reusable functions; **must be in Python path** for imports to work

## Key Data Flows & Patterns

### Data Preprocessing Pipeline
```python
from master_function import data_preprocessing
x_train, y_train, x_test, y_test = data_preprocessing(data, num_lags, train_test_split=0.80)
```
- Takes 1D time series, creates lagged inputs (features) and shifted outputs (targets)
- Always use after differencing/stationarity checks (see Chapter 3 patterns)
- Returns train/test split in sequential order (temporal integrity maintained)

### Multi-Period Forecasting (MPF)
Two competing approaches in Chapter 9:
- **Direct MPF**: Single model predicts entire forecast horizon at once; use for long-term stability
- **Recursive MPF**: Single-step model applied iteratively; use when num_lags > forecast_horizon

Functions: `recursive_mpf()` and `direct_mpf()` handle architecture-specific reshaping (MLP vs LSTM)

### Model Architectures
- **MLP**: `Dense(neurons, activation='relu')` → Dense(1); input shape = num_lags
- **LSTM**: Input reshaped to `(-1, num_lags, 1)` before LSTM layer; add Dense layers after for feature extraction
- **CNN**: See Chapter 8 for Conv1D patterns
- All models use Keras Sequential API with Adam optimizer and MSE loss

## Critical Developer Workflows

### Running Scripts
1. Ensure `master_function.py` is in your Python interpreter's directory (Spyder: top-right, VS Code: project root)
2. Financial data fetched via `pandas_datareader` (FRED API) or `MetaTrader5` terminal (requires MT5 installed)
3. COT (Commitments of Traders) data via `cot_reports` library; some scripts import by year range (2010-2023)

### Evaluation Metrics
- **Directional Accuracy**: `calculate_directional_accuracy(y_predicted, y_test)` - checks sign agreement
- **Model Bias**: `model_bias()` - captures systematic over/under-prediction
- Standard regression: RMSE via `sklearn.metrics.mean_squared_error`

## Project-Specific Conventions

1. **Hyperparameter naming**: `num_lags`, `num_neurons_in_hidden_layers`, `num_epochs`, `batch_size`, `train_test_split`
2. **Data stationarity**: Patterns in Chapter 1/3 show differencing (`data.diff()`) before modeling
3. **Plotting standard**: `plot_train_test_values(window=100, train_window=50, y_train, y_test, y_predicted)` - overlays training/test/predicted on shared window
4. **Forecasting threshold**: Zero out small predictions with `forecasting_threshold(predictions, threshold)`
5. **Array reshaping**: Always validate dimensions for LSTM (-1, num_lags, 1) vs Dense (-1, num_lags)

## Integration Points & External Dependencies

- **Data Sources**: FRED (economic data), MT5 (forex), COT reports (positioning data)
- **Core Libraries**: Keras/TensorFlow, scikit-learn, NumPy, pandas, matplotlib
- **Input Validation**: MetaTrader5 requires terminal running; pandas_datareader needs internet; COT data subject to availability
- **Time Zone Handling**: MT5 functions use `pytz.timezone("Europe/Paris")` for UTC conversion

## When Adding New Models
1. Preprocess data with `data_preprocessing()` maintaining 80/20 split
2. If LSTM, reshape to 3D before training
3. Store model predictions in consistent format: `y_predicted_train`, `y_predicted` (test)
4. Always report directional accuracy + RMSE for consistency with existing chapters
5. Reference `master_function.py` functions rather than reimplementing preprocessing logic
