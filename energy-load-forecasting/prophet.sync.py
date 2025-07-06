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
# # Energy Load Forecasting with Prophet
#
# Context:<br />
# A company creates an app that helps minimize energy costs for users,
# figuring out when to use batteries and when to use the grid.<br />
# To do this, they need to forecast the energy load for the next 2 days.

# %% [markdown]
# ## Imports

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from matplotlib.dates import HourLocator, DateFormatter

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.make_holidays import make_holidays_df
from neuralprophet import NeuralProphet
# xgboost
from xgboost import XGBRegressor
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import os
import json
from multiprocessing import Pool, cpu_count

# %%
# Verify cuda is available
import torch

print(torch.__version__)
print(torch.cuda.is_available())      # Should be True
# print(torch.cuda.get_device_name(0))  # Should show your GPU name

# %%
RANDOM_STATE = 42

# %% [markdown]
# ## Visualization Setup

# %%
sns.color_palette('tab10')

# %%
sns.color_palette('muted')

# %%
sns.color_palette('colorblind')

# %%
blue = sns.color_palette('colorblind')[0]
green = sns.color_palette('colorblind')[2]

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
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelsize': 'medium',
        }
    )

# %% [markdown]
# ## Load Data

# %%
# Source: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
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

# %%
pjmw_df

# %%
pjmw_df.info()

# %% [markdown]
# Are there any duplicates in the datetime column?

# %%
pjmw_df['datetime'].duplicated().sum()

# %%
pjmw_df[pjmw_df['datetime'].duplicated(keep=False)]

# %% [markdown]
# These duplicate entries are really close to each other in terms of
# consumptions, they can be dropped.

# %%
pjmw_df = pjmw_df.drop_duplicates(subset='datetime', keep='last').reset_index(drop=True)

# %% [markdown]
# Are there any missing dates in the datetime column?

# %%
pjmw_df.sort_values('datetime', inplace=True)
full_range = pd.date_range(start=pjmw_df['datetime'].min(), end=pjmw_df['datetime'].max(), freq='H')

# %%
missing_dates_df = pd.DataFrame({'datetime': full_range.difference(pjmw_df['datetime'])})
missing_dates_df

# %% [markdown]
# That's peculiar. Could this be a timezone issue perhaps? Let's see which timezone the datetime column is in.

# %% [markdown]
# I couldn't find the timezone information in the dataset, so let's interpolate these
# missing dates and proceed with the analysis.

# %%
pjmw_df = pjmw_df.set_index('datetime').reindex(full_range).rename_axis('datetime').reset_index()
pjmw_df['mw'] = pjmw_df['mw'].interpolate(method='linear')

# %%
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
        color=blue,
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
# Some sanity checks

# %%
train_val_test_df['ds'].min(), train_val_test_df['ds'].max()

# %%
train_val_test_df.isna().sum()

# %%
train_val_test_df['ds'].duplicated().sum()

# %%
cuttoff_date = train_val_test_df['ds'].max() - pd.Timedelta(days=365 * 3)  # 3 years ago
train_val_test_df = train_val_test_df[train_val_test_df['ds'] >= cuttoff_date].reset_index(drop=True)

# %%
train_val_test_df['day_of_week'] = train_val_test_df['ds'].dt.day_name()
train_val_test_df['day_of_week'].value_counts().sort_index()

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

# %% [markdown]
# Some sanity checks on the splits.

# %%
train_df['day_of_week'].value_counts().sort_index()

# %%
val_df['day_of_week'].value_counts().sort_index()

# %%
test_df['day_of_week'].value_counts().sort_index()

# %%
# Define the horizon for the Prophet model to predict
# horizon_hours = 7 * 24  # in hours
horizon_hours = 2 * 24  # in hours

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
INTERVAL_WIDTH = 0.95
ci_str = f"{int(INTERVAL_WIDTH * 100)}%"

print(f"Confidence interval: {ci_str}")


# %%
def objective_prophet(trial):
    params = {
        "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
        "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10.0, log=True),
        "holidays_prior_scale":    trial.suggest_float("holidays_prior_scale", 0.01, 10.0, log=True),
        "seasonality_mode":        trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
        "changepoint_range":       trial.suggest_float("changepoint_range", 0.8, 0.95),
        "daily_seasonality":   trial.suggest_categorical("daily_seasonality", [True]),
        "weekly_seasonality": trial.suggest_categorical("weekly_seasonality", [True]),
        "yearly_seasonality": trial.suggest_categorical("yearly_seasonality", [True]),
        "interval_width": INTERVAL_WIDTH
    }

    model = Prophet(**params)

    model.add_seasonality(
        name='hourly',
        period=24,
        fourier_order=8  # Controls flexibility of hourly pattern
    )

    model.add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=5
    )

    # Add holidays (optional)
    holidays = make_holidays_df(year_list=range(2015, 2026), country='US')

    model.add_country_holidays(country_name='US')

    model.fit(train_df)

    # # Forecast on validation
    # future_df = model.make_future_dataframe(periods=len(val_df), freq='H')
    # forecast_df = model.predict(future_df.tail(len(val_df)))

    # # # Match forecast to actuals
    # y_pred = forecast_df['yhat'].values
    # y_true = val_df['y'].values

    # return mean_absolute_error(y_true, y_pred)

    # Cross-validation
    df_cv = cross_validation(
        model,
        initial='730 days',
        period=f'{horizon_hours // 2} hours',
        horizon=f'{horizon_hours} hours',
        parallel="processes"   # Optional: parallelizes across cores
    )

    # Evaluate using MAE (can use other metrics if needed)
    df_p = performance_metrics(df_cv, rolling_window=1)
    mae = df_p['mae'].mean()

    return mae


# %%
# %%time

timeout_in_seconds = 60 * 10  # 10 minutes

study = optuna.create_study(direction="minimize")
# study.optimize(objective_prophet, n_trials=50, timeout=timeout_in_seconds, n_jobs=4)
# No benefit from parallelizing here, as Prophet is already parallelized internally for cross-validation.
# study.optimize(objective_prophet, n_trials=50, timeout=timeout_in_seconds)
study.optimize(objective_prophet, n_trials=50)

best_prophet_params = study.best_params.copy()

print("Best parameters:", study.best_params)
print("Best validation MAE:", study.best_value)

# %%
print("Best parameters:", study.best_params)
print("Best validation MAE:", study.best_value)

# %%
# create model directory if it doesn't exist

if not os.path.exists('models'):
    os.makedirs('models')

with open('models/best_prophet_params.json', 'w') as f:
    json.dump(best_prophet_params, f, indent=2)

# %%
with open('models/best_prophet_params.json', 'r') as f:
    best_prophet_params = json.load(f)

# %%
best_prophet_model = Prophet(**best_prophet_params)
best_prophet_model.fit(train_df)

# %%
# Predict on training set with model
pjmw_val_fcst = best_prophet_model.predict(
        df=val_df.reset_index().rename(columns={'Datetime':'ds'})
        )

# Plot the forecast
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
fig = best_prophet_model.plot(pjmw_val_fcst, ax=ax)
plt.show()

# %% [markdown]
# Since we care about the next two days, let's limit the plot to that horizon.

# %%
# Filter forecast for only 2 days
start_date = pjmw_val_fcst['ds'].min()
end_date = start_date + pd.Timedelta(days=2)
pjmw_val_fcst_2d = pjmw_val_fcst[pjmw_val_fcst['ds'] < end_date]

fig, ax = plt.subplots(1)
fig.set_figheight(5)
fig.set_figwidth(15)
best_prophet_model.plot(pjmw_val_fcst_2d, ax=ax)

ax.set_title('Prophet Forecast for PJMW Load (Next 2 Days)')
ax.set_xlim([
    start_date - pd.Timedelta(days=31),
    end_date
    ])

# ax.set_ylim([0, ax.get_ylim()[1]])  # Keep y-axis limits consistent

ax.set_xlabel('Date')
ax.set_ylabel('MW')

plt.show()

# %%

# %%
# Filter forecast for only 2 days
start_date = pjmw_val_fcst['ds'].min()
end_date = start_date + pd.Timedelta(days=2)
pjmw_val_fcst_2d = pjmw_val_fcst[pjmw_val_fcst['ds'] < end_date]

fig, ax = plt.subplots(1)
fig.set_figheight(5)
fig.set_figwidth(15)
best_prophet_model.plot(pjmw_val_fcst_2d, ax=ax)

ax.set_title('Prophet Forecast for PJMW Load (Next 2 Days)')
ax.set_xlim([
    start_date,
    end_date
    ])

# ax.set_ylim([0, ax.get_ylim()[1]])  # Keep y-axis limits consistent

ax.set_xlabel('Date')
ax.set_ylabel('MW')

x_axis_labels = pjmw_val_fcst_2d['ds'].dt.strftime('%H')

ax.set_xticks(pjmw_val_fcst_2d['ds'])
ax.set_xticklabels(x_axis_labels, ha='right')

ax.xaxis.set_major_locator(HourLocator(interval=3))

plt.show()

# %%
pjmw_val_fcst_2d.head(1).T

# %%
val_df.head(1).T

# %% [markdown]
# ## Evaluation
# We will use a rolling origin evaluation, feeding all data before the cutoff
# date to the model, predicting for the horizon, and then extending the training
# data by the horizon. We will repeat this process to get a more accurate
# measurement of the models performance, using the validation set.

# %%
# %%time

val_data = val_df.reset_index().rename(columns={'Datetime': 'ds'})[['ds', 'y']]
train_data = train_df.copy()

step_days = 2
forecast_horizon = pd.Timedelta(days=step_days)
start_time = val_data['ds'].min()
end_time = val_data['ds'].max()

jobs = []
current_time = start_time

while current_time + forecast_horizon <= end_time:
    jobs.append(current_time)
    current_time += forecast_horizon

def forecast_one_step(current_time):
    try:
        # Subset observed validation data
        observed_val = val_data[val_data['ds'] < current_time]
        rolling_train = pd.concat([train_data, observed_val], ignore_index=True)

        # Define prediction window
        predict_window = val_data[
            (val_data['ds'] >= current_time) &
            (val_data['ds'] < current_time + forecast_horizon)
        ]

        if predict_window.empty:
            return None  # Skip

        model = Prophet()
        model.fit(rolling_train)

        future = predict_window[['ds']]
        forecast = model.predict(future)

        return {
            'forecast': pd.merge(forecast, predict_window, on='ds')
        }
    except Exception as e:
        print(f"Error at {current_time}: {e}")
        return None

with Pool(processes=cpu_count()) as pool:
    results = pool.map(forecast_one_step, jobs)

# %%
all_forecasts = [res['forecast'] for res in results if res is not None]
forecast_df = pd.concat(all_forecasts, ignore_index=True)

# %%
forecast_df

# %%
forecast_df['mae'] = np.abs(forecast_df['yhat'] - forecast_df['y'])
forecast_df['mape'] = np.abs(forecast_df['yhat'] - forecast_df['y']) / forecast_df['y']

forecast_df[['y', 'yhat', 'mae', 'mape']].describe()

# %% [markdown]
# We are about 10% off on average in our predictions. Let's use the interval provided by prophet.

# %% [markdown]
# Let's see how far off we are using the confidence intervals provided by prophet.

# %%
forecast_df['yhat_lower_adj'] = forecast_df[['yhat_lower', 'y']].max(axis=1)
forecast_df['yhat_upper_adj'] = forecast_df[['yhat_upper', 'y']].min(axis=1)
forecast_df['yhat_lower_adj_mae'] = np.abs(forecast_df['yhat_lower_adj'] - forecast_df['y'])
forecast_df['yhat_upper_adj_mae'] = np.abs(forecast_df['yhat_upper_adj'] - forecast_df['y'])
forecast_df['yhat_adj_mae'] = forecast_df['yhat_lower_adj_mae'] + forecast_df['yhat_upper_adj_mae']
forecast_df['yhat_adj_mape'] = forecast_df['yhat_adj_mae'] / forecast_df['y']
forecast_df['ci_range'] = forecast_df['yhat_upper'] - forecast_df['yhat_lower']
forecast_df['ci_range_pct'] = forecast_df['ci_range'] / forecast_df['y'] * 100

forecast_df[['mae', 'yhat_adj_mae', 'mape', 'yhat_adj_mape', 'ci_range_pct', 'yhat_lower_adj_mae', 'yhat_upper_adj_mae']].describe()

# %% [markdown]
# How often does our confidence interval contain y?

# %%
pct = ((forecast_df['yhat_lower'] <= forecast_df['y']) & (forecast_df['yhat_upper'] >= forecast_df['y'])).mean()
print(f"Percentage of actuals within confidence interval: {pct:.1%}")

# %% [markdown]
# Around 73% of the time our confidence interval contains the actual value.

# %% [markdown]
# Let's see the distribution of the absolute errors, for yhat and adjusted yhat.

# %%
fig, ax = plt.subplots(figsize=(12, 6))

sns.histplot(
    data=forecast_df,
    x='mae',
    bins=50,
    ax=ax,
    color=blue,
    label='Prediction Absolute Error'
)

sns.histplot(
    data=forecast_df,
    x='yhat_adj_mae',
    bins=50,
    ax=ax,
    color=green,
    label=f'{ci_str} Confidence Interval Absolute Error'
)

ax.set_title('Distribution of Absolute Errors')
ax.set_xlabel('Absolute Error in MW')
ax.set_ylabel('Frequency')
ax.legend(title='Error Type')
plt.tight_layout()
plt.show()

# %% [markdown]
# Let's see the distribution of the absolute errors, excluding the zero errors.

# %%
fig, ax = plt.subplots(figsize=(12, 6))

viz_df_mae = forecast_df[forecast_df['mae'] != 0]
viz_df_mae_adj = forecast_df[forecast_df['yhat_adj_mae'] != 0]

sns.histplot(
    data=viz_df_mae,
    x='mae',
    bins=50,
    ax=ax,
    color=blue,
    label='Prediction Absolute Error'
)

sns.histplot(
    data=viz_df_mae_adj,
    x='yhat_adj_mae',
    bins=50,
    ax=ax,
    color=green,
    label=f'{ci_str} Confidence Interval Absolute Error'
)

ax.set_title('Distribution of Absolute Errors (excluding accurate predictions)')
ax.set_xlabel('Absolute Error in MW')
ax.set_ylabel('Frequency')
ax.legend(title='Error Type')
plt.tight_layout()
plt.show()

# %% [markdown]
# When using the confidence interval instead of the predicted value from prophet, our error is reduced considerably. If the range provided as the confidence interval is acceptable, we can use it to have a more accurate estimate of the actual consumption.

# %%
forecast_df['mean_error'] = forecast_df['yhat'] - forecast_df['y']

forecast_df['mean_error_adj'] = np.where(
    forecast_df['yhat_lower'] >= forecast_df['y'],
    forecast_df['yhat_lower_adj'] - forecast_df['y'],
    forecast_df['yhat_upper_adj'] - forecast_df['y']
)

# %%
cols = ['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper', 'mean_error', 'mean_error_adj']

# %%
forecast_df[forecast_df['mean_error_adj'] == 0][cols].head(3).T

# %%
forecast_df[forecast_df['mean_error_adj'] > 0][cols].head(3).T

# %%
forecast_df[forecast_df['mean_error_adj'] < 0][cols].head(3).T

# %%
forecast_df[forecast_df['mean_error'] == 0][cols].head(3).T

# %%
forecast_df[forecast_df['mean_error'] > 0][cols].head(3).T

# %%
forecast_df[forecast_df['mean_error'] < 0][cols].head(3).T

# %%
forecast_df.sort_values('mean_error').head(5).T

# %%
forecast_df[forecast_df['yhat_upper'] < forecast_df['y']].T

# %% [markdown]
# Let's see the distribution of the non-absolute errors, for yhat and adjusted yhat.

# %%
fig, ax = plt.subplots(figsize=(12, 6))

sns.histplot(
    data=forecast_df,
    x='mae',
    bins=50,
    ax=ax,
    color=blue,
    label='Prediction Absolute Error'
)

sns.histplot(
    data=forecast_df,
    x='yhat_adj_mae',
    bins=50,
    ax=ax,
    color=green,
    label=f'{ci_str} Confidence Interval Absolute Error'
)

ax.set_title('Distribution of Absolute Errors (excluding accurate predictions)')
ax.set_xlabel('Absolute Error in MW')
ax.set_ylabel('Frequency')
ax.legend(title='Error Type')
plt.tight_layout()
plt.show()

# %% [markdown]
# Let's see same distribution excluding the zero errors.

# %%
fig, ax = plt.subplots(figsize=(12, 6))

viz_df_mean_error = forecast_df[forecast_df['mean_error'] != 0]
viz_df_mean_error_adj = forecast_df[forecast_df['mean_error_adj'] != 0]

sns.histplot(
    data=viz_df_mean_error,
    x='mean_error',
    bins=50,
    ax=ax,
    color=blue,
    label='Prediction Error'
)

sns.histplot(
    data=viz_df_mean_error_adj,
    x='mean_error_adj',
    bins=50,
    ax=ax,
    color=green,
    label=f'{ci_str} Confidence Interval Error'
)

ax.set_title('Distribution of Non-Absolute Errors (exclusing accurate predictions)')
ax.set_xlabel('Non-Absolute Error in MW')
ax.set_ylabel('Frequency')
ax.legend(title='Error Type')
plt.tight_layout()
plt.show()

# %% [markdown]
# Are we predicting consumption on all hours of the day with the same accuracy?

# %%
# plot the hourly MAE, with hue for adjusted yhat and yhat
fig, ax = plt.subplots(figsize=(12, 6))

viz_df = forecast_df.copy()[['ds', 'mae', 'yhat_adj_mae']]
viz_df = pd.melt(
    viz_df,
    id_vars=['ds'],
    value_vars=['mae', 'yhat_adj_mae'],
    var_name='error_type',
    value_name='error'
)

viz_df['hour'] = viz_df['ds'].dt.hour

viz_df['error_type'] = viz_df['error_type'].map({
    'mae': 'Prediction MAE',
    'yhat_adj_mae': f'{ci_str} Confidence Interval MAE'
    })

sns.barplot(
    data=viz_df,
    x='hour',
    y='error',
    hue='error_type',
    ax=ax,
    palette=[blue, green],
    errorbar=None,
    dodge=False
    )

ax.set_title('Hourly MAE of Prophet Forecast')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Mean Absolute Error (MAE) in MW')
xticks = np.arange(0, 24, 1)
ax.set_xticks(xticks)
ax.set_xticklabels(
        [f'{x:02d}' for x in xticks]
        )

# set legend title
ax.legend(title='Error type', loc='upper left')


plt.tight_layout()
plt.show()

# %% [markdown]
# We are a little more accurate during the night hours but not by much.

# %% [markdown]
# Let's plot the hourly consumption to see if there are any patterns.

# %%
fig, ax = plt.subplots(figsize=(12, 6))

viz_df = val_df.copy()
viz_df['hour'] = viz_df['ds'].dt.hour
viz_df['day_of_week'] = viz_df['ds'].dt.day_name()
viz_df['day_of_week'] = pd.Categorical(
    viz_df['day_of_week'],
    categories=[
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ],
    ordered=True
)
viz_df = viz_df.groupby(['hour', 'day_of_week'], as_index=False).mean() \
        .rename(columns={'y': 'mw'})


sns.lineplot(
    data=viz_df,
    x='hour',
    y='mw',
    ax=ax,
    linewidth=1.5,
    hue='day_of_week'
)
ax.set_title('Average Hourly Consumption of PJMW Load')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('MW')
xticks = np.arange(0, 24, 1)
ax.set_xticks(xticks)
ax.set_xticklabels(
        [f'{x:02d}' for x in xticks]
        )

ax.set_ylim([0, ax.get_ylim()[1]])  # Keep y-axis limits consistent
plt.tight_layout()
plt.show()

# %% [markdown]
#
# As expected, consumption is on average a little lower during the night hours and we can focus on recharging batteries during those hours with enough confidence that our predictions are accurate and that estimated costs are minimized. We can also keep in mind that during midday, even though on average consumption is not at its highest, we have less accurate predictions and could make our users use the grid resulting in higher costs. If we could prioritize times of day when average consumption is low **and** our predictions are as accurate as possible, we will consistently minimize costs for our users.

# %% [markdown]
# ## Conclusion
# We used Facebook's Prophet to predict energy consumption for the next two
# days. Our model returned a prediction about energy consumption for each hour
# of the next 2 days, as well as a confidence interval (95%) for the actual
# consumption. Using the prediction of the model as input, an algorithm can
# optimize the use of batteries to minimize energy costs within the next two
# days.
# The model is about 10% off in its predictions (MAPE ~= 0.1). Using the
# confidence intervals we can reduce the error to about 2% on average. The
# confidence interval contains the actual energy consumption about 73% of the
# time for the validation set.

# %%
