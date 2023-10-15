import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import tensorflow as tf
#import rpy2
import warnings
#import logging
import optuna
import streamlit as st
import seaborn as sns
from numpy.linalg import norm
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_percentage_error as mape, mean_absolute_error as mae, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV, KFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.stats import boxcox, kruskal, randint
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.deterministic import DeterministicProcess
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.stattools import durbin_watson
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
#from rpy2 import robjects
#from rpy2.robjects.packages import importr, data
#from rpy2.robjects import pandas2ri, r
#from rpy2.robjects.functions import SignatureTranslatedFunction
from sklearn.linear_model import LinearRegression
#from pmdarima.arima import auto_arima
from sklearn.model_selection import TimeSeriesSplit
optuna.logging.set_verbosity(optuna.logging.WARNING)
#logging.getLogger('optuna').setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

def minmax(x, y=None):
    if y is None:
        y = x
    return (x - y.min()) / (y.max() - y.min())

def minmax_rev(x, y):
    return x * (y.max() - y.min()) + y.min()

def feature_engineering(data, lags=3):
    if isinstance(data, pd.Series):
        data = data.to_frame()
    target_col = data.columns[-1]

    for i in range(1, lags + 1):
        data[f'lag_{i}'] = data[target_col].shift(i)
    data['rolling_mean_3'] = data[target_col].rolling(window=3).mean()
    data['rolling_std_3'] = data[target_col].rolling(window=3).std()
    data['rolling_max_3'] = data[target_col].rolling(window=3).max()

    data.dropna(inplace=True)
    return data

def objective(trial, X_train, y_train, X_valid, y_valid):
    params = {
        'n_estimators': trial.suggest_int("n_estimators", 30, 100),
        'learning_rate': trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int("max_depth", 3, 5),
        'subsample': trial.suggest_float("subsample", 0.5, 1.0),
        'max_features': trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        'validation_fraction': len(X_valid) / (len(X_train) + len(X_valid)),
        'n_iter_no_change': 6,
        'tol': 0.01,
        'warm_start': True
    }
    model = GradientBoostingRegressor(**params)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    return mse(y_valid, y_pred)

def determine_D(data, s=12, max_D=3):
    data = np.array(data).flatten()
    d_data = np.diff(data)
    D = 0
    result = adfuller(d_data)

    if result[1] > 0.05:
        for D in range(1, max_D + 1):
            s_diff_data = np.diff(d_data, n=s * D)
            result = adfuller(s_diff_data)
            if result[1] <= 0.05:
                break

    return D

def determine_seasonality_order(series, lags=20, max_diff=2):
    if isinstance(series, np.ndarray):
        temp_series = series.copy().flatten()
    elif isinstance(series, pd.Series):
        temp_series = series.values.copy()
    else:
        raise ValueError("Input should be a numpy ndarray or pandas Series.")

    d = 0
    while d < max_diff and adfuller(temp_series)[1] > 0.05:
        temp_series = np.diff(temp_series)
        d += 1

    # Determine significance level for ACF
    n = len(temp_series)
    threshold = 1.645 / np.sqrt(n)  # For a 90% significance level

    acf_values = acf(temp_series, nlags=lags)

    significant_lags = [i for i, value in enumerate(acf_values[1:], start=1) if abs(value) > threshold]

    differences = [significant_lags[i + 1] - significant_lags[i] for i in range(len(significant_lags) - 1)]
    seasonality = Counter(differences).most_common(1)[0][0] if differences else 0

    return seasonality

def determine_p_P(series, s=12, max_lags=20):
    pacf_vals = pacf(series, nlags=max_lags)

    # Determine p from the PACF
    p = 0
    for i in range(1, s):  # we are excluding the zero lag
        if pacf_vals[i] > 0.2:  # arbitrary threshold, adjust as needed
            p = i
        else:
            break

    # Determine P from the PACF at seasonal lags
    P = 0
    for i in range(s, max_lags, s):
        if pacf_vals[i] > 0.2:  # arbitrary threshold, adjust as needed
            P = i // s
        else:
            break

    return p, P

def best_SARIMA_order(train_data, valid_data, max_d=2, max_q=5, max_Q=3, seasonal=True):
    d = 0
    try:
        temp_series = train_data.copy().flatten()
    except AttributeError:
        temp_series = train_data.copy()

    while d < max_d and adfuller(temp_series)[1] > 0.05:
        temp_series = np.diff(temp_series)
        d += 1

    s = determine_seasonality_order(train_data) if seasonal else 0
    if s <= 1:
        st.write("Inferred seasonality was 1 or less. Defaulting to seasonality of 12.")
        s = 12
    # s = 12
    D = determine_D(train_data) if seasonal else 0
    p, P = determine_p_P(train_data, s)

    # Grid search for best p, q, P, Q
    best_order = None
    best_seasonal_order = None
    best_bic = float("inf")

    for q in range(max_q + 1):
        for Q in range(max_Q + 1):
            try:
                model = SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
                res = model.fit()

                valid_pred = res.predict(start=len(train_data), end=len(train_data) + len(valid_data) - 1)
                valid_mse = mse(valid_data, valid_pred)

                current_bic = len(valid_data) * np.log(valid_mse) + q * np.log(len(train_data))

                if current_bic < best_bic:
                    best_bic = current_bic
                    best_order = (p, d, q)
                    best_seasonal_order = (P, D, Q, s)
                    best_model_res = res

            except Exception as e:
                st.write(f"Error for SARIMA({p},{d},{q})x({P},{D},{Q},{s}): {e}")
                continue

    st.write(f"Best SARIMA order is {best_order} with seasonal order {best_seasonal_order} and BIC = {best_bic:.2f}")
    return best_order, best_seasonal_order, best_model_res

def best_ARIMA_order(train_data, valid_data, max_p=5, max_d=1, max_q=5):
    best_order = None
    best_bic = float('inf')
    best_model_res = None

    d = 0
    while d <= max_d:
        adf_test = adfuller(train_data, maxlag=d, autolag='AIC')
        if adf_test[1] <= 0.05 or d == max_d:
            break
        d += 1

    for p in range(max_p + 1):
        for q in range(1, max_q + 1):
            try:
                model = ARIMA(train_data, order=(p, d, q))
                res = model.fit()

                valid_pred = res.predict(start=len(train_data), end=len(train_data) + len(valid_data) - 1)
                valid_mse = mse(valid_data, valid_pred)

                current_bic = len(valid_data) * np.log(valid_mse) + q * np.log(len(train_data))

                if current_bic < best_bic:
                    best_bic = current_bic
                    best_order = (p, d, q)
                    best_model_res = res

            except Exception as e:
                st.write(f"Error for ARIMA({p},{d},{q}): {e}")
                continue

    st.write(f"Best ARIMA order is {best_order} with BIC = {best_bic:.2f}")
    return best_order, best_model_res

def best_MA_order(train_data, valid_data, max_q=10):
    best_q = None
    best_bic = float('inf')
    best_model_res = None

    for q in range(1, max_q + 1):
        try:
            model = ARIMA(train_data, order=(0, 0, q))
            res = model.fit()

            valid_pred = res.predict(start=len(train_data), end=len(train_data) + len(valid_data) - 1)
            valid_mse = mse(valid_data, valid_pred)

            current_bic = len(valid_data) * np.log(valid_mse) + q * np.log(len(train_data))

            if current_bic < best_bic:
                best_bic = current_bic
                best_q = q
                best_model_res = res
        except Exception as e:
            st.write(f"Error for q = {q}: {e}")
            continue

    st.write(f"Best q is {best_q} with BIC = {best_bic:.2f}")
    return best_q, best_model_res

def best_AR_order(train_data, valid_data, max_p=10):
    best_p = None
    best_bic = float('inf')
    best_model_res = None

    for p in range(1, max_p + 1):
        try:
            model = ARIMA(train_data, order=(p, 0, 0))
            res = model.fit()

            valid_pred = res.predict(start=len(train_data), end=len(train_data) + len(valid_data) - 1)
            valid_mse = mse(valid_data, valid_pred)

            current_bic = len(valid_data) * np.log(valid_mse) + p * np.log(len(train_data))

            if current_bic < best_bic:
                best_bic = current_bic
                best_p = p
                best_model_res = res
        except Exception as e:
            st.write(f"Error for p = {p}: {e}")
            continue

    st.write(f"Best p is {best_p} with BIC = {best_bic:.2f}")
    return best_p, best_model_res

def determine_polynomial_degree(X_train, y_train, X_val, y_val, max_degree=4):
    best_degree = 1
    best_score = float('inf')

    for degree in range(1, max_degree + 1):
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train, y_train)

        valid_pred = model.predict(X_val)
        score = mse(y_val, valid_pred)

        if score < best_score:
            best_score = score
            best_degree = degree

    return best_degree

def detect_trend(ts, alpha=0.005, trend_type='linear', r2_threshold=0.6):
    n = len(ts)
    X = np.column_stack((np.ones(n), np.arange(n)))
    ts = np.array(ts)

    if trend_type == 'linear':
        pass
    elif trend_type == 'exponential':
        ts = np.log(ts)
    elif trend_type == 'polynomial':
        X = np.column_stack((X, X[:, 1] ** 2))
    else:
        raise ValueError("Invalid trend_type. Use 'linear', 'polynomial', or 'exponential'.")

    model = sm.OLS(ts, X).fit()

    f_statistic_p_value = model.f_pvalue
    if f_statistic_p_value > alpha:
        return False

    trend_p_value = model.pvalues[-1]
    if trend_p_value >= alpha:
        return False

    if model.rsquared < r2_threshold:
        return False

    bp_test = sm.stats.diagnostic.het_breuschpagan(model.resid, model.model.exog)
    if bp_test[1] <= alpha:
        return False

    return True

def check_noise(ts, threshold=0.05):
    total_variance = np.var(ts)

    seasonal_trend = sm.tsa.seasonal_decompose(ts)
    detrended_series = ts - seasonal_trend.seasonal - seasonal_trend.trend

    noise_variance = np.var(detrended_series)
    noise_ratio = noise_variance / total_variance
    if noise_ratio < threshold:
        return False
    else:
        return True  # CHANGED

def detect_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    outlier_condition = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
    return outlier_condition.any()

def non_linear_dependencies(data):
    X = np.arange(len(data)).reshape(-1, 1)
    y = data.values

    model = LinearRegression()
    model.fit(X, y)
    predicted = model.predict(X)

    residuals = y - predicted

    threshold = np.std(data) * 0.5  # This is arbitrary and might need adjustments based on specific datasets
    if np.std(residuals) > threshold:
        return True

    return False

def calculate_score(row, weights):
    score = 0
    for metric, weight in weights.items():
        score += weight * row[metric]
    return score

weights = {
    'R^2': -0.25,  # Still significant, but not overwhelming
    'SUM(e(k)^2)': 0.1,  # Reduced, as it's similar to MSE
    'DW': -0.05,  # Kept minimal, more of a diagnostic tool
    'MSE': 0.15,  # Balanced among error metrics
    'RMSE': 0.2,  # Slightly higher as it's more commonly referenced
    'MAE': 0.15,  # Similar to MSE, but more robust to outliers
    # 'MAPE': 0.05,          # Reduced due to sensitivity issues
    'U': 0.05  # Kept minimal, lesser-used metric
}

def rolling_forecast_sarima(train_data, test_data, order, seasonal_order, exog_oos=None):
    history = list(train_data)
    predictions = []
    for t in range(len(test_data)):
        model = ARIMA(history, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        output = model_fit.forecast(steps=1, exog=exog_oos) if exog_oos else model_fit.forecast(steps=1)
        yhat = output[0]
        predictions.append(yhat)
        history.append(test_data[t])
    return predictions

results_df = pd.DataFrame(columns=['model_name', 'R^2', 'SUM(e(k)^2)', 'DW', 'MSE', 'RMSE', 'MAE', 'U', 'model_object'])

def u_coef(y_true, y_pred):
    return mse(y_true, y_pred) ** 1 / 2 / (norm(y_true) + norm(y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def model_stats(train_true, test_true, result, model_name, model_type='ARIMA', exog_oos=None, target_col=None):
    global results_df

    if model_type in ['AutoReg', 'MovingAverage', 'ARIMA', 'SARIMA']:
        try:
            st.write(result.summary())
        except AttributeError as e:
            st.write("Error: The object doesn't have a summary method:", e)

        train_true_normalized = minmax(train_true)
        test_true_normalized = minmax(test_true, train_true)

        p = result.model.order[0]
        d = result.model.order[1]
        q = result.model.order[2]
        P, D, Q, s = result.model.seasonal_order

        if model_type == 'AutoReg':
            train_true_reduced = train_true[p:]
            train_pred_normalized = result.predict(start=p, end=len(train_true) - 1)
            train_pred = minmax_rev(train_pred_normalized, train_true)
            r2_val = r2_score(train_true[p:], train_pred)

        elif model_type == 'MovingAverage':
            train_true_reduced = train_true[q:]
            train_pred_normalized = result.predict(start=q, end=len(train_true) - 1)
            train_pred = minmax_rev(train_pred_normalized, train_true)
            r2_val = r2_score(train_true[q:], train_pred)
            # r2_val = r2_score(train_true_normalized[q:], train_pred)

        elif model_type == 'ARIMA':
            start_point = max(p, d + q)
            train_true_reduced = train_true[start_point:]
            train_pred_normalized = result.predict(start=start_point, end=len(train_true) - 1, typ='levels')
            train_pred = minmax_rev(train_pred_normalized, train_true)
            r2_val = r2_score(train_true[start_point:], train_pred)

        elif model_type == 'SARIMA':
            start_point = max(p + d + q, P + D + Q + s - 1)
            train_true_reduced = train_true[start_point:]
            train_pred_normalized = result.predict(start=start_point, end=len(train_true) - 1, typ='levels')
            train_pred = minmax_rev(train_pred_normalized, train_true)
            r2_val = r2_score(train_true[start_point:], train_pred)
            # test_pred = rolling_forecast_sarima(train_true, test_true, (p,d,q), (P,D,Q,s), exog_oos)

        test_pred_normalized = result.predict(start=len(train_true_reduced),
                                              end=len(train_true_reduced) + len(test_true) - 1, typ='levels',
                                              exog_oos=exog_oos)
        test_pred = minmax_rev(test_pred_normalized, train_true)

        residuals = train_true_reduced - train_pred

        train_pred = train_pred.fillna(0)
        test_pred = test_pred.fillna(0)

        new_stats = pd.DataFrame({
            'model_name': [model_name],
            'R^2': [r2_val],
            'SUM(e(k)^2)': [(result.resid ** 2).sum()],
            'DW': [durbin_watson(residuals)],
            'RMSE': [np.sqrt(mse(test_true, test_pred))],
            'MAE': [mae(test_true, test_pred)],
            'MSE': [mse(test_true, test_pred)],
            # 'MAPE': [mape(test_true, test_pred)],
            'U': [u_coef(test_true, test_pred)],
            'model_object': result
        })

    elif model_type == 'ETS':
        train_true_normalized = minmax(train_true)
        test_true_normalized = minmax(test_true, train_true)

        train_pred_normalized = result.predict(start=0, end=len(train_true) - 1)
        test_pred_normalized = result.forecast(steps=len(test_true))

        residuals = train_true_normalized - train_pred_normalized

        train_pred = minmax_rev(train_pred_normalized, train_true)
        test_pred = minmax_rev(test_pred_normalized, train_true)

    elif model_type in ['LinearRegression', 'PolynomialRegression']:
        train_true_normalized = minmax(train_true)
        test_true_normalized = minmax(test_true, train_true)

        X_train = np.arange(len(train_true)).reshape(-1, 1)
        X_test = np.arange(len(train_true), len(train_true) + len(test_true)).reshape(-1, 1)

        train_pred_normalized = result.predict(X_train)
        test_pred_normalized = result.predict(X_test)

        residuals = train_true_normalized - train_pred_normalized

        train_pred = minmax_rev(train_pred_normalized, train_true)
        test_pred = minmax_rev(test_pred_normalized, train_true)

    elif model_type == 'ExponentialModel':
        train_true_normalized = minmax(train_true)
        test_true_normalized = minmax(test_true, train_true)

        X_train = np.arange(len(train_true)).reshape(-1, 1)
        X_test = np.arange(len(train_true), len(train_true) + len(test_true)).reshape(-1, 1)

        train_log_pred_normalized = result.predict(X_train)
        test_log_pred_normalized = result.predict(X_test)

        train_pred_normalized = np.expm1(train_log_pred_normalized)
        test_pred_normalized = np.expm1(test_log_pred_normalized)

        residuals = train_true_normalized - train_pred_normalized

        train_pred = minmax_rev(train_pred_normalized, train_true)
        test_pred = minmax_rev(test_pred_normalized, train_true)

    elif model_type in ['GradientBoosting']:
        X_train_normalized = minmax(train_true.drop(target_col, axis=1))
        X_test_normalized = minmax(test_true.drop(target_col, axis=1), train_true.drop(target_col, axis=1))

        train_pred_normalized = result.predict(X_train_normalized)
        test_pred_normalized = result.predict(X_test_normalized)

        residuals = train_true[target_col] - train_pred_normalized

        train_pred = minmax_rev(train_pred_normalized, train_true[target_col])
        test_pred = minmax_rev(test_pred_normalized, train_true[target_col])

        new_stats = pd.DataFrame({
            'model_name': [model_name],
            'R^2': [r2_score(train_true[target_col][-len(train_pred):], train_pred)],
            'SUM(e(k)^2)': [(np.array(residuals) ** 2).sum()],
            'DW': [durbin_watson(residuals)],
            'MSE': [mse(test_true[target_col], test_pred)],
            'RMSE': [np.sqrt(mse(test_true[target_col], test_pred))],
            'MAE': [mae(test_true[target_col], test_pred)],
            # 'MAPE': [mape(test_true[target_col], test_pred)],
            'U': [u_coef(test_true[target_col], test_pred)],
            'model_object': [result]
        })

    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    if model_type not in ['GradientBoosting', 'ARIMA', 'AR']:
        new_stats = pd.DataFrame({
            'model_name': [model_name],
            'R^2': [r2_score(train_true[-len(train_pred):], train_pred)],
            'SUM(e(k)^2)': [(np.array(residuals) ** 2).sum()],
            'DW': [durbin_watson(residuals)],
            'MSE': [mse(test_true, test_pred)],
            'RMSE': [np.sqrt(mse(test_true, test_pred))],
            'MAE': [mae(test_true, test_pred)],
            # 'MAPE': [mape(test_true, test_pred)],
            'U': [u_coef(test_true, test_pred)],
            'model_object': [result]
        })

    if model_type in ['AutoReg', 'MovingAverage', 'ARIMA', 'SARIMA']:
        plt.figure(figsize=(18, 6), dpi=200)
        sns.lineplot(x=np.arange(len(train_true_reduced)), y=train_true_reduced, color='blue', label='Train True')
        sns.lineplot(x=np.arange(len(train_true_reduced)), y=train_pred, color='orange', label='Train Predicted')
        sns.lineplot(x=np.arange(len(train_true_reduced), len(train_true_reduced) + len(test_true)), y=test_true,
                     color='green', label='Test True')
        sns.lineplot(x=np.arange(len(train_true_reduced), len(train_true_reduced) + len(test_true)), y=test_pred,
                     color='red', label='Test Predicted')
        plt.title(f"Performance of {model_type} Model")
        st.pyplot(plt)
    elif model_type == 'GradientBoosting':
        plt.figure(figsize=(18, 6), dpi=200)
        sns.lineplot(x=np.arange(len(train_true[target_col])), y=train_true[target_col], color='blue',
                     label='Train True')
        sns.lineplot(x=np.arange(len(train_true[target_col])), y=train_pred, color='orange', label='Train Predicted')
        sns.lineplot(x=np.arange(len(train_true[target_col]), len(train_true[target_col]) + len(test_true[target_col])),
                     y=test_true[target_col], color='green', label='Test True')
        sns.lineplot(x=np.arange(len(train_true[target_col]), len(train_true[target_col]) + len(test_true[target_col])),
                     y=test_pred, color='red', label='Test Predicted')
        plt.title(f"Performance of {model_type} Model")
        st.pyplot(plt)
    else:
        plt.figure(figsize=(18, 6), dpi=200)
        sns.lineplot(x=np.arange(len(train_true)), y=train_true, color='blue', label='Train True')
        sns.lineplot(x=np.arange(len(train_true)), y=train_pred, color='orange', label='Train Predicted')
        sns.lineplot(x=np.arange(len(train_true), len(train_true) + len(test_true)), y=test_true, color='green',
                     label='Test True')
        sns.lineplot(x=np.arange(len(train_true), len(train_true) + len(test_true)), y=test_pred, color='red',
                     label='Test Predicted')
        plt.title(f"Performance of {model_type} Model")
        st.pyplot(plt)

    results_df = pd.concat([results_df, new_stats], ignore_index=True)
    st.write(results_df.drop(columns='model_object'))

class TimeSeriesAnalyzer:

        def __init__(self, data):
            self.raw_data = data
            self.data = data
            self.features = {}
            self.potential_models = []
            self.best_model_name = None
            self.fitted_model = None
            self.original_data = data.copy()
            self.data = data.copy()

        def analyze_features(self):

            adf_result = adfuller(self.data)
            self.features['stationary'] = adf_result[1] <= 0.05

            decomposition = seasonal_decompose(self.data)
            seasonal_component = decomposition.seasonal
            residuals = decomposition.resid.dropna()
            iqr = seasonal_component.quantile(0.75) - seasonal_component.quantile(0.25)
            seasonality_strength = iqr / seasonal_component.std()
            self.features['strong_seasonality'] = seasonality_strength > 0.64
            self.features['weak_seasonality'] = 0 < seasonality_strength <= 0.64

            n = len(self.data)
            threshold = 1.645 / np.sqrt(n)
            self.features['correlation'] = bool(np.max(acf(self.data)[1:]) > threshold)

            #shapiro_test = stats.shapiro(self.data)
            shapiro_test = stats.shapiro(residuals)
            self.features['normal_distribution'] = shapiro_test[1] > 0.05

            self.features['stochastic_trend'] = not self.features['stationary']

            self.features['linear_trend'] = detect_trend(self.data, trend_type='linear')

            self.features['exponential_trend'] = detect_trend(self.data, trend_type='exponential')

            self.features['polynomial_trend'] = detect_trend(self.data, trend_type='polynomial')

            trend_types_present = any([
                self.features.get('linear_trend', False),
                self.features.get('exponential_trend', False),
                self.features.get('polynomial_trend', False),
                self.features.get('stochastic_trend', False)
            ])

            if trend_types_present:
                self.features['trend'] = True
            else:
                self.features['trend'] = False

            #fft_values = np.abs(np.fft.fft(self.data))
            #threshold = np.percentile(fft_values, 99)
            #self.features['complex_patterns'] = np.sum(fft_values > threshold) > 2

            self.features['long_series'] = len(self.data) > 1000  # TODO adjust threshold

            self.features['volatile'] = self.data.pct_change().std() > 0.05  # TODO adjust threshold

            rolling_mean = self.data.rolling(window=12).mean()
            self.features['change_points'] = (
                                                     rolling_mean.diff() > rolling_mean.std()).sum() > 2  # Arbitrary thresholds

            complex_features_present = any([
                self.features.get('volatile', False),
                self.features.get('non_linear_dependencies', False),
                self.features.get('outliers', False),
                self.features.get('change_points', False)
            ])

            if complex_features_present:
                self.features['has_complex_feature'] = True
            else:
                self.features['has_complex_feature'] = False

            #n = len(self.data)
            #threshold = 1.96 / np.sqrt(n)
            #autocorrelations = acf(self.data, nlags=20)
            #self.features['embedding_dim'] = (np.abs(autocorrelations) > threshold).sum()  # Count significant lags

            self.features['significant_noise'] = check_noise(self.data)

            self.features['non_linear_dependencies'] = non_linear_dependencies(self.data)

            self.features['outliers'] = detect_outliers(self.data)

            return self.features

        def select_models(self):
            self.potential_models = set()

            if (self.features['trend']) and not (
                    self.features['has_complex_feature']
                    or self.features['significant_noise']
                    or self.features['correlation']
                    or self.features['stochastic_trend']
                    or sum(self.features.values()) > 10):  # Adjust threshold
                self.potential_models.add('ETS')

            if (self.features['weak_seasonality'] and self.features['correlation'] and self.features['trend']) \
                    or (
                    self.features['trend'] and self.features['weak_seasonality'] and not self.features['stationary']) \
                    and not (self.features['has_complex_feature'] or self.features['significant_noise']):
                self.potential_models.add('ARIMA')

            if ((self.features['strong_seasonality'] and self.features['correlation'] and self.features['trend']) \
                or (self.features['trend'] and self.features['weak_seasonality'] and not self.features['stationary'])) \
                    and not (self.features['has_complex_feature'] or self.features['significant_noise']):
                self.potential_models.add('SARIMA')

            if self.features['stationary'] and self.features['significant_noise'] and not (
                    self.features['has_complex_feature'] or self.features['strong_seasonality'] or self.features[
                'stochastic_trend']):
                self.potential_models.add('MovingAverage')

            if self.features['stationary'] and self.features['correlation'] and not (
                    self.features['has_complex_feature'] or self.features['strong_seasonality'] or self.features[
                'stochastic_trend']):
                self.potential_models.add('AutoReg')

            if self.features['linear_trend'] and self.features['normal_distribution']:
                self.potential_models.add('LinearRegression')

            if self.features['polynomial_trend'] and self.features['normal_distribution']:
                self.potential_models.add('PolynomialRegression')

            if self.features['exponential_trend'] and self.features['normal_distribution']:
                self.potential_models.add('ExponentialModel')

            if self.features['has_complex_feature'] or self.features['long_series'] or self.features['correlation'] or \
                    self.features['significant_noise']:
                self.potential_models.add('GradientBoosting')

            if not self.potential_models:
                self.potential_models.add('GradientBoosting')
                raise ValueError("No suitable model could be determined for the given features!")

            self.potential_models = list(self.potential_models)
            return self.potential_models

        def train_and_validate(self, model_name):

            train_size = int(len(self.data) * 0.6)
            valid_size = train_size + int(len(self.data) * 0.2)
            train_data_raw = self.data[:train_size]
            valid_data_raw = self.data[train_size:valid_size]
            test_data_raw = self.data[valid_size:]

            train_data = minmax(train_data_raw)
            valid_data = minmax(valid_data_raw, train_data_raw)
            test_data = minmax(test_data_raw, train_data_raw)
            combined_data_normalized = pd.concat([train_data, valid_data])

            if model_name == 'ETS':
                possible_trends = ['add', 'mul', None] if self.features['trend'] else [None]
                possible_seasonal = ['add', 'mul'] if self.features['strong_seasonality'] else [None]
                # seasonal_periods = 12
                seasonal_periods = determine_seasonality_order(self.data) if self.features['strong_seasonality'] else [
                    None]
                if seasonal_periods == 1 and self.features['strong_seasonality']:
                    st.write("Inferred seasonality was 1 or less. Defaulting to seasonality of 12.")
                    seasonal_periods = 12
                best_model = None
                best_score = float('inf')

                for trend_type in possible_trends:
                    for seasonal_type in possible_seasonal:
                        if train_data.min() <= 0 and (trend_type == 'mul' or seasonal_type == 'mul'):
                            st.write(
                                f"Skipping configuration trend={trend_type}, seasonal={seasonal_type} due to non-positive values in data.")
                            continue
                        try:
                            model = ExponentialSmoothing(train_data, trend=trend_type, seasonal=seasonal_type,
                                                         seasonal_periods=seasonal_periods)
                            res = model.fit()

                            valid_pred = res.predict(start=len(train_data), end=len(train_data) + len(valid_data) - 1)
                            valid_mse = mse(valid_data, valid_pred)

                            if valid_mse < best_score:
                                best_model = res
                                best_score = valid_mse
                                best_trend_type = trend_type
                                best_seasonal_type = seasonal_type

                        except Exception as e:
                            st.write(f"Error with configuration trend={trend_type}, seasonal={seasonal_type}: {e}")
                            continue

                combined_data_normalized = pd.concat([train_data, valid_data])
                best_ets_model = ExponentialSmoothing(combined_data_normalized, trend=best_trend_type,
                                                      seasonal=best_seasonal_type, seasonal_periods=seasonal_periods)
                best_ets_model_res = best_ets_model.fit()

                model_description = f'ETS(Trend={best_model.model.trend}, Seasonal={best_model.model.seasonal}, Periods={seasonal_periods})'
                model_stats(train_data_raw, test_data_raw, best_ets_model_res, model_description, model_type='ETS')

            elif model_name == 'SARIMA':
                max_q = 4
                max_Q = 4
                max_d = 2

                best_order, best_seasonal_order, best_model_res = best_SARIMA_order(train_data, valid_data, max_d,
                                                                                    max_q, max_Q)

                best_sarima_model = SARIMAX(combined_data_normalized, order=best_order,
                                            seasonal_order=best_seasonal_order)
                best_sarima_model_res = best_sarima_model.fit()

                model_stats(train_data_raw, test_data_raw, best_sarima_model_res,
                            f'SARIMA{best_order}x{best_seasonal_order}', model_type='SARIMA')

            elif model_name == 'ARIMA':
                max_p = 5
                max_q = 5
                max_d = 2
                best_order, best_model_res = best_ARIMA_order(train_data, valid_data, max_p, max_d, max_q)

                best_arima_model = ARIMA(combined_data_normalized, order=best_order)
                best_arima_model_res = best_arima_model.fit()

                model_stats(train_data_raw, test_data_raw, best_arima_model_res, f'ARIMA{best_order}',
                            model_type='ARIMA')

            elif model_name == 'AutoReg':
                max_p = 10
                best_p, best_model_res = best_AR_order(train_data, valid_data, max_p)

                best_ar_model = ARIMA(combined_data_normalized, order=(best_p, 0, 0))
                best_ar_model_res = best_ar_model.fit()

                model_stats(train_data_raw, test_data_raw, best_ar_model_res, f'AR({best_p})', model_type='AutoReg')
                # forecast_and_plot(f'AR({best_p})', combined_data_normalized, test_data_raw)

            elif model_name == 'MovingAverage':
                max_q = 10
                best_q, best_model_res = best_MA_order(train_data, valid_data, max_q)
                best_ma_model = ARIMA(combined_data_normalized, order=(0, 0, best_q))
                best_ma_model_res = best_ma_model.fit()

                model_stats(train_data_raw, test_data_raw, best_ma_model_res, f'MA({best_q})',
                            model_type='MovingAverage')

            elif model_name == 'LinearRegression':
                model = LinearRegression()
                X = np.arange(len(combined_data_normalized)).reshape(-1, 1)
                fitted = model.fit(X, combined_data_normalized)
                model_stats(train_data_raw, test_data_raw, fitted, 'LinearRegression', model_type='LinearRegression')

            elif model_name == 'PolynomialRegression':
                X_train = np.arange(len(train_data)).reshape(-1, 1)
                X_val = np.arange(len(train_data), len(train_data) + len(valid_data)).reshape(-1, 1)

                max_degree = 4
                best_degree = determine_polynomial_degree(X_train, train_data, X_val, valid_data, max_degree)

                best_pr_model = make_pipeline(PolynomialFeatures(best_degree), LinearRegression())
                best_pr_model_res = best_pr_model.fit(np.vstack((X_train, X_val)), combined_data_normalized)

                model_stats(train_data_raw, test_data_raw, best_pr_model_res, 'PolynomialRegression',
                            model_type='PolynomialRegression')

            elif model_name == 'ExponentialModel':
                train_log = np.log1p(combined_data_normalized)
                model = LinearRegression()
                X = np.arange(len(combined_data_normalized)).reshape(-1, 1)
                fitted = model.fit(X, train_log)
                metrics = model_stats(train_data_raw, test_data_raw, fitted, 'ExponentialModel',
                                      model_type='ExponentialModel')

            elif model_name == 'GradientBoosting':
                combined_data_fe = feature_engineering(combined_data_normalized.copy())
                target_col = combined_data_fe.columns[-1]

                X_combined = combined_data_fe.drop(target_col, axis=1)
                y_combined = combined_data_fe[target_col]

                best_params = None
                best_score = float('inf')
                kfold = KFold(n_splits=5, shuffle=True, random_state=42)

                for train_idx, valid_idx in kfold.split(X_combined):
                    X_train_fold, X_valid_fold = X_combined.iloc[train_idx], X_combined.iloc[valid_idx]
                    y_train_fold, y_valid_fold = y_combined.iloc[train_idx], y_combined.iloc[valid_idx]

                    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
                    n_trials = 20
                    study.optimize(
                        lambda trial: objective(trial, X_train_fold, y_train_fold, X_valid_fold, y_valid_fold),
                        n_trials=n_trials)

                    if study.best_value < best_score:
                        best_score = study.best_value
                        best_params = study.best_params

                best_model = GradientBoostingRegressor(**best_params)
                best_model.fit(X_combined, y_combined)

                test_data_fe = feature_engineering(test_data.copy())
                X_test, y_test = test_data_fe.drop(target_col, axis=1), test_data_fe[target_col]
                metrics = model_stats(pd.concat([X_combined, y_combined], axis=1), pd.concat([X_test, y_test], axis=1),
                                      best_model, model_name, model_type=model_name, target_col=target_col)
            else:
                raise ValueError(f"Unknown model name: {model_name}")

        def plot_results(self, start_index, end_index):
            # Ensure the data has been indexed
            assert start_index < end_index, "Invalid indexes"

            # Get the actual and predicted values
            actual_values = self.data[start_index:end_index]
            try:
                predicted_values = self.fitted_model.predict(start=start_index, end=end_index - 1)
            except:
                # For models that don't have a typical predict method
                predicted_values = self.fitted_model.forecast(steps=end_index - start_index)

            # Plot actual vs predicted values
            plt.figure(figsize=(14, 7))
            plt.plot(actual_values, label="Actual Values", color='blue')
            plt.plot(predicted_values, label="Predicted Values", color='red', linestyle='--')
            plt.title(f"Actual vs Predicted Values for {self.best_model_name}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.show()

        def compare_models(self, weights):
            results_df['score'] = results_df.apply(lambda row: calculate_score(row, weights), axis=1)

            results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            results_df['score'].fillna(1e10, inplace=True)  # using a large value
            # results_df['score'].fillna(0, inplace=True)
            results_df['score'] = pd.to_numeric(results_df['score'], errors='coerce')

            best_model_row = results_df.iloc[results_df['score'].idxmin()]
            self.best_model_name = best_model_row['model_name']
            self.fitted_model = best_model_row['model_object']

            st.write(results_df.drop(columns=['model_object', 'score']))  # CHANGE

            st.markdown("<div style='background-color: #e6e6e6; padding: 10px; border-radius: 5px;'>", unsafe_allow_html=True)
            st.markdown("### **Best Model**")
            st.markdown(f"<h3>Model Name: {self.best_model_name}</h3>", unsafe_allow_html=True)
            st.markdown(f"**Score:** {best_model_row['score']}")
            st.markdown("</div>", unsafe_allow_html=True)

            try:
                st.write(self.fitted_model.summary())
            except AttributeError:
                st.write("No summary method for this model.")

        def run_analysis(self):
            st.subheader("Analyzing Features...")
            self.analyze_features()
            st.write("Features analyzed successfully.\n")

            with st.expander("Features"):
                for feature, value in self.features.items():
                    if value:
                        st.write(f"{feature}")

            st.subheader("\nSelecting Potential Models...")
            potential_models = self.select_models()
            st.markdown("**Selected Models:**")
            for model in potential_models:
                st.text(f"- {model}")
            st.subheader("\nTraining & Validating Models...")
            results = []
            for model_name in potential_models:
                st.markdown(f"<span style='color: blue; font-weight: bold;'>Processing {model_name}...</span>", unsafe_allow_html=True)
                with st.expander(f"{model_name}", expanded=False):
                    try:
                        self.train_and_validate(model_name)
                        st.markdown(f"<div style='color: green;'><b>{model_name}</b> processed successfully.</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f"<div style='color: red;'>Error processing <b>{model_name}</b>: {str(e)}</div>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("<div style='color: teal; text-align: center;'>Comparing Models...</div>", unsafe_allow_html=True)
            self.compare_models(weights)
            st.write("")
            st.markdown("<h1 style='text-align: center; '>Analysis Completed!</h1>", unsafe_allow_html=True)