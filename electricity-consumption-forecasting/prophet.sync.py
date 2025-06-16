# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: myenv
#     language: python
#     name: myenv
# ---

# %% [markdown]
# ## Imports

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

from prophet import Prophet
from neuralprophet import NeuralProphet
# xgboost
from xgboost import XGBRegressor
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# %%
# Verify cuda is a available
import torch

print(torch.__version__)
print(torch.cuda.is_available())      # Should be True
# print(torch.cuda.get_device_name(0))  # Should show your GPU name

# %%
RANDOM_STATE = 42

# %% [markdown]
# ## Visualization Setup

# %%
font_path = '/usr/share/fonts/noto_sans_mono/NotoSansMono_SemiCondensed-SemiBold.ttf'
font_prop = fm.FontProperties(fname=font_path)

sns.set_theme(
    style='darkgrid',
    context='notebook',
    font=font_prop.get_name(),
    rc={
        'font.weight': 'semibold',
        'axes.labelweight': 'semibold',
        'axes.titlesize': 'large',
        'axes.titleweight': 'semibold',
        'axes.labelsize': 'medium',
        }
    )

# %% [markdown]
# ## Load Data

# %%
pjmw_df = pd.read_csv('data/PJMW_hourly.csv')

# %%
pjmw_df.shape

# %%
pjmw_df.isna().sum()

# %%
pjmw_df.info()

# %%
pjmw_df.head()

# %%
pjmw_df.rename(columns={'Datetime': 'datetime', 'PJMW_MW': 'mw'}, inplace=True)

# %%
pjmw_df['datetime'] = pd.to_datetime(pjmw_df['datetime'])
pjmw_df['year'] = pjmw_df['datetime'].dt.year
pjmw_df['month'] = pjmw_df['datetime'].dt.month
pjmw_df['day'] = pjmw_df['datetime'].dt.day
pjmw_df['hour'] = pjmw_df['datetime'].dt.hour
pjmw_df['day_of_week'] = pjmw_df['datetime'].dt.dayofweek
pjmw_df['day_of_year'] = pjmw_df['datetime'].dt.dayofyear
pjmw_df['week_of_year'] = pjmw_df['datetime'].dt.isocalendar().week
pjmw_df['quarter'] = pjmw_df['datetime'].dt.quarter

# %% [markdown]
# ## EDA

# %%
fig, ax = plt.subplots(figsize=(18, 6))

sns.lineplot(
        data=pjmw_df,
        x='datetime',
        y='mw',
        ax=ax,
        color='blue',
        linewidth=1.5
        )

ax.set_title('PJMW Hourly Load')
ax.set_xlabel('Datetime')
ax.set_ylabel('MW')
ax.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
pjmw_df.head()

# %% [markdown]
# Let's see how each of the time features correlates with the load (MW).

# %%
x_vars = [
    'year',
    'month',
    'day',
    'hour',
    'day_of_week',
    'day_of_year',
    'week_of_year',
    'quarter'
]

fig, axes = plt.subplots(4, 2, figsize=(16, 16))  # 4 rows, 2 columns
axes = axes.flatten()

for i, var in enumerate(x_vars):
    sns.scatterplot(data=pjmw_df, x=var, y='mw', ax=axes[i])
    axes[i].set_title(f'{var} vs mw')

plt.suptitle('Grid of Pair Plots for PJMW Data', fontsize=18, y=1.02, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
train_val_test_df = pjmw_df.copy()[['datetime', 'mw']]. \
        rename(columns={'datetime': 'ds', 'mw': 'y'}). \
        sort_values(by='ds'). \
        reset_index(drop=True)

train_val_test_df.head()

# %%
train_val_test_df.info()

# %% [markdown]
# ### Some sanity checks

# %%
train_val_test_df['ds'].min(), train_val_test_df['ds'].max()

# %%
train_val_test_df.isna().sum()

# %%
train_val_test_df['ds'].value_counts().value_counts()

# %%
value_counts = train_val_test_df['ds'].value_counts()
value_counts[value_counts > 1].sort_values(ascending=False)

# %%
train_val_test_df['ds'].duplicated().sum()

# %%
train_val_test_df[train_val_test_df['ds'].duplicated(keep=False)].sort_values(by='ds')

# %% [markdown]
# Let's drop duplicates and keep the last entry

# %%
train_val_test_df = train_val_test_df.drop_duplicates(
        subset='ds',
        keep='last'
        ).reset_index(drop=True)

# %%
total_days = (train_val_test_df['ds'].max() - train_val_test_df['ds'].min()).days

train_days = int(0.7 * total_days)
val_days   = int(0.15 * total_days)
test_days  = total_days - train_days - val_days

train_end = train_val_test_df['ds'].min() + pd.Timedelta(days=train_days)
val_end   = train_end + pd.Timedelta(days=val_days)

train_df = train_val_test_df[train_val_test_df['ds'] <= train_end]
val_df   = train_val_test_df[(train_val_test_df['ds'] > train_end) & (train_val_test_df['ds'] <= val_end)]
test_df  = train_val_test_df[train_val_test_df['ds'] > val_end]

# %%
# print size, min and max dates of each set
print(f"Train set:      {train_df.shape}, {train_df['ds'].min()} - {train_df['ds'].max()}")
print(f"Validation set: {val_df.shape}, {val_df['ds'].min()} - {val_df['ds'].max()}")
print(f"Test set:       {test_df.shape}, {test_df['ds'].min()} - {test_df['ds'].max()}")

# %%
# print the number of NAs in each set

print(f"Train set NAs:      {train_df.isna().sum().sum()}")
print(f"Validation set NAs: {val_df.isna().sum().sum()}")
print(f"Test set NAs:       {test_df.isna().sum().sum()}")


# %%
def objective_prophet(trial):
    params = {
        "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
        "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10.0, log=True),
        "holidays_prior_scale":    trial.suggest_float("holidays_prior_scale", 0.01, 10.0, log=True),
        "seasonality_mode":        trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
        "changepoint_range":       trial.suggest_float("changepoint_range", 0.8, 0.95),
    }

    model = Prophet(**params)
    model.fit(train_df)

    # Forecast on validation
    future_df = model.make_future_dataframe(periods=len(val_df), freq='H')
    forecast_df = model.predict(future_df.tail(len(val_df)))

    # Match forecast to actuals
    y_pred = forecast_df['yhat'].values
    y_true = val_df['y'].values

    return mean_absolute_error(y_true, y_pred)


# %%
# %%time

study = optuna.create_study(direction="minimize")
study.optimize(objective_prophet, n_trials=50, timeout=600, n_jobs=4)
best_prophet_params = study.best_params.copy()

print("Best parameters:", study.best_params)
print("Best validation MAE:", study.best_value)

# %%
best_prophet_model = Prophet(**best_prophet_params)
best_prophet_model.fit(train_df)

# %%
model = best_prophet_model
future_df = model.make_future_dataframe(periods=len(val_df), freq='H')
forecast_df = model.predict(future_df)
forecast_df_val = forecast_df[forecast_df['ds'].isin(val_df['ds'])]

# %%
len(val_df)

# %%
len(future_df)

# %%
len(forecast_df)

# %%
len(forecast_df_val)

# %%
forecast_df_val.info()

# %%
val_df.info()

# %%
forecast_df_val['ds'].duplicated().sum()

# %%
val_df['ds'].duplicated().sum()

# %%
set(val_df['ds'].astype(str).values) - set(forecast_df_val['ds'].astype(str).values)

# %%
forecast_df_val['ds'].max()

# %%
forecast_df_val['ds'].min()


# %%

# %%

# %%

# %%
def evaluate_model(model, val_df):
    # Forecast on validation
    future_df = model.make_future_dataframe(periods=len(val_df), freq='H')
    forecast_df = model.predict(future_df.tail(len(val_df)))

    # Match forecast to actuals
    y_pred_val = forecast_df['yhat'].values
    y_true_val = val_df['y'].values

    val_mae = mean_absolute_error(y_true_val, y_pred_val)

    return val_mae, y_pred_val, y_true_val

def plot_results(y_true_val, y_pred_val):
    plt.figure(figsize=(14, 7))

    # Validation set
    plt.plot(y_true_val, label='Actual (Validation)', color='blue', alpha=0.7)
    plt.plot(y_pred_val, label='Predicted (Validation)', color='orange', linestyle='--', alpha=0.7)
    plt.title('Validation Set Predictions')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# %%
val_mae, y_pred_val, y_true_val = evaluate_model(
    best_prophet_model, val_df
)

val_mean = val_df['y'].mean()
print(f"Validation MAE: {val_mae:.4f}")
print(f"Validation MAE / Mean: {val_mae / val_mean:.4f}")

# %%
# Plot results
plot_results(y_true_val, y_pred_val)

# %%

# %%
future_df = model.make_future_dataframe(periods=len(val_df), freq='H')
forecast_df = model.predict(future_df.tail(len(val_df)))
forecast_df

# %%
forecast_df.head(1).T

# %%
forecast_df.columns

# %%
forecast_df['y'] = val_df['y'].values
forecast_df['residual'] = forecast_df['yhat'] - forecast_df['y']

# %%
residual_train_val_test_df = forecast_df.copy().drop(columns=['y'])

residual_train_val_test_df['day'] = residual_train_val_test_df['ds'].dt.day
residual_train_val_test_df['month'] = residual_train_val_test_df['ds'].dt.month
residual_train_val_test_df['year'] = residual_train_val_test_df['ds'].dt.year
residual_train_val_test_df['hour'] = residual_train_val_test_df['ds'].dt.hour
residual_train_val_test_df['day_of_week'] = residual_train_val_test_df['ds'].dt.dayofweek
residual_train_val_test_df['day_of_year'] = residual_train_val_test_df['ds'].dt.dayofyear
residual_train_val_test_df['week_of_year'] = residual_train_val_test_df['ds'].dt.isocalendar().week
residual_train_val_test_df['quarter'] = residual_train_val_test_df['ds'].dt.quarter

residual_train_val_test_df.drop(columns=['ds'], inplace=True)

X = residual_train_val_test_df.drop(columns=['residual', 'yhat'])
y = residual_train_val_test_df['residual']

# Split: train (60%), val (20%), test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=RANDOM_STATE, shuffle=False
        )

X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=RANDOM_STATE,
        shuffle=False
        )


# %%

# %%
def objective_xgboost(trial):
    params = {
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'random_state': RANDOM_STATE
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)  # RMSE

    return rmse


# %%
# %%time
study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(objective_xgboost, n_trials=50, timeout=600)

print("Best parameters for XGBoost:", study_xgb.best_params)
print("Best validation RMSE for XGBoost:", study_xgb.best_value)

# %%
best_xgb_model = XGBRegressor(**study_xgb.best_params)
best_xgb_model.fit(X_train_val, y_train_val)

# %%
corrected_future_df = model.make_future_dataframe(periods=len(test_df), freq='H')
forecast_df_test = model.predict(corrected_future_df.tail(len(test_df)))


# %%

# %%
def objective_neural_prophet(trial):
    # Define hyperparameters to optimize
    params = {
        'n_lags': trial.suggest_int('n_lags', 5, 30),
        'n_forecasts': trial.suggest_int('n_forecasts', 1, 7),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'epochs': trial.suggest_int('epochs', 50, 150),
        'batch_size': trial.suggest_categorical('batch_size', [512, 1024, 2048]),
        'loss_func': trial.suggest_categorical('loss_func', ['Huber', 'MAE', 'MSE']),
    }

    try:
        # Create and train model
        model = NeuralProphet(
            n_lags=params['n_lags'],
            n_forecasts=params['n_forecasts'],
            learning_rate=params['learning_rate'],
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            loss_func=params['loss_func'],
            collect_metrics=False
        )

        # Fit model on training data
        model.fit(train_df, freq='H')

        # Make predictions on val data
        future = model.make_future_dataframe(
            train_df,
            periods=len(val_df),
            n_historic_predictions=True
        )
        forecast = model.predict(future)

        # Extract val predictions
        val_pred = forecast.tail(len(val_df))['yhat1'].values
        val_actual = val_df['y'].values

        # Calculate and return MAE
        mae = mean_absolute_error(val_actual, val_pred)
        return mae

    except Exception as e:
        # Return high error for failed trials
        print(f"Trial failed: {e}")
        return float('inf')


# %%
def check_train_val_gap(train_df, val_df):
    last_train = train_df['ds'].max()
    first_val = val_df['ds'].min()
    gap_days = (first_val - last_train).days

    print(f"Last training date: {last_train}")
    print(f"First validation date: {first_val}")
    print(f"Gap between train and val: {gap_days} days")

    if gap_days != 1:
        print(f"⚠️  WARNING: Expected 1 day gap, found {gap_days} days")
        return False
    else:
        print("✅ No gap detected")
        return True

# Check your data
check_train_val_gap(train_df, val_df)

# %%
test_model = NeuralProphet()
test_model.fit(train_df, freq='H')

# %%
# %%time
study = optuna.create_study(direction="minimize")
study.optimize(objective_neural_prophet, n_trials=50, timeout=600)

print("Best parameters for NeuralProphet:", study.best_params)
print("Best validation MAE for NeuralProphet:", study.best_value)

# %%
best_model = NeuralProphet(
    n_lags=study.best_params['n_lags'],
    n_forecasts=study.best_params['n_forecasts'],
    learning_rate=study.best_params['learning_rate'],
    epochs=study.best_params['epochs'],
    batch_size=study.best_params['batch_size'],
    d_hidden=study.best_params['d_hidden'],
    loss_func=study.best_params['loss_func']
)

best_model.fit(train_data, freq='H')

# Final evaluation
future = best_model.make_future_dataframe(
    train_data,
    periods=len(test_data),
    n_historic_predictions=True
)
final_forecast = best_model.predict(future)

final_pred = final_forecast.tail(len(test_data))['yhat1'].values
final_mae = mean_absolute_error(test_data['y'].values, final_pred)

print(f"Final model MAE: {final_mae:.4f}")

# %%
# Optional: Plot results
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(test_data['y'].values, label='Actual', alpha=0.7)
    plt.plot(final_pred, label='Predicted', alpha=0.7)
    plt.title('NeuralProphet Predictions (Optimized)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

except ImportError:
    print("Matplotlib not available for plotting")

# Show parameter importance
print("\nParameter Importance:")
importance = optuna.importance.get_param_importances(study)
for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{param}: {imp:.3f}")

# %%
