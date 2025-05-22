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
# # Adjusting Monthly Targets to a Daily Level
#
# Targets are usually provided on a monthly, quarterly, or yearly level.
# However, when tracking performance, we want to see if we are on track to meet
# our targets on a daily basis, not at the end of the month, quarter or year, in
# order to make adjustments in time.
# <br />
# This notebook demonstrates how to adjust monthly targets to a daily level
# using the Walmart sales dataset. It utilizes features such as holidays,
# day of the week, and other calendar features to create a daily target
# dataset, based on the monthly targets and on the historical sales data.

# %% [markdown]
# ## Imports

# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
import optuna
from datetime import datetime, timedelta
import humanize
from sklearn import metrics
from sklearn.tree import plot_tree
import joblib

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import matplotlib.ticker as ticker

# %% [markdown]
# ## Visualization Setup and Helper Functions

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


# %%
# Returns the x coordinates of a text element on a given axis of a given
# figure.
# Used to position elements on the canvas
# Returns object with attributes:
#   x0 coordinate of the text element
#   x1 coordinate of the text element
#   y0 coordinate of the text element
#   y1 coordinate of the text element
def get_text_coordinates(text_element, ax, fig):
        x0 = text_element.get_window_extent(fig.canvas.get_renderer()).x0
        x1 = text_element.get_window_extent(fig.canvas.get_renderer()).x1
        y0 = text_element.get_window_extent(fig.canvas.get_renderer()).y0
        y1 = text_element.get_window_extent(fig.canvas.get_renderer()).y1
        return {
                 'x0': round(ax.transData.inverted().transform_point((x0, 0))[0], 2),
                 'x1': round(ax.transData.inverted().transform_point((x1, 0))[0], 2),
                 'y0': round(ax.transData.inverted().transform_point((0, y0))[1], 2),
                 'y1': round(ax.transData.inverted().transform_point((0, y1))[1], 2)
               }


# %% [markdown]
# ## Load Data

# %%
# Source https://www.kaggle.com/datasets/devarajv88/walmart-sales-dataset
pd.read_csv('data/sales_train_validation.csv')

# %%
pd.read_csv('data/sales_train_evaluation.csv')

# %%
pd.read_csv('data/calendar.csv')

# %%
pd.read_csv('data/sell_prices.csv')

# %%
prices_df = pd.read_csv('data/sell_prices.csv')
sales_df = pd.read_csv('data/sales_train_validation.csv')

# %%
d_cols = [f'd_{i}' for i in range(1, 1914)]


# %% [markdown]
# Melt the sales dataframe to have a long format with one row per item per day.
# Then join with the calendar dataframe to get the date and other calendar
# features, join to get the prices and sum up to get the GMV.

# %%
def melt_sales_df(df):
    melted_df = pd.melt(
            df,
            id_vars=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
            value_vars=[f'd_{i}' for i in range(1, 1914)],
            var_name='day', value_name='sales'
            )

    return melted_df


# %%
melted_sales_df = melt_sales_df(sales_df)

# %%
melted_sales_df

# %%
calendar_df = pd.read_csv('data/calendar.csv')
calendar_df

# %%
# Merge the melted sales dataframe with the calendar dataframe
merged_df = pd.merge(
    melted_sales_df,
    calendar_df[['d', 'date', 'wday', 'month', 'year', 'wm_yr_wk', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']],
    left_on='day',
    right_on='d'
)

merged_df.drop(columns=['d'], inplace=True)
merged_df

# %%
# Merge the merged dataframe with the prices dataframe
merged_df_with_prices = pd.merge(
    merged_df,
    prices_df[['store_id', 'item_id', 'sell_price', 'wm_yr_wk']],
    on=['store_id', 'item_id', 'wm_yr_wk']
)

merged_df_with_prices

# %%
merged_df_with_prices.isna().sum()

# %%
merged_df_with_prices['gmv'] = merged_df_with_prices['sales'] * merged_df_with_prices['sell_price']

# %%
agg_df = merged_df_with_prices.groupby('date').agg({
    'gmv': 'sum',
    'event_name_1': lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan,
    'event_type_1': lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan,
    'event_name_2': lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan,
    'event_type_2': lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan,
    'month': 'first',
    'year': 'first',
    'wday': 'first'
    }).reset_index()

agg_df

# %%
agg_df.isna().sum()

# %%
agg_df[~agg_df['event_name_1'].isna()]

# %%
agg_df.groupby('year')['date'].nunique()

# %%
monthly_df = agg_df.groupby(['year', 'month'])['gmv'].sum().reset_index()
monthly_df

# %%
agg_df_with_monthly = pd.merge(
    agg_df,
    monthly_df,
    on=['year', 'month'],
    suffixes=('', '_monthly')
)

agg_df_with_monthly['monthly_gmv_share'] = agg_df_with_monthly['gmv'] / agg_df_with_monthly['gmv_monthly']

# %%
agg_df_with_monthly

# %%
agg_df_with_monthly.drop(columns=['gmv', 'gmv_monthly'], inplace=True)

# %%
agg_df_with_monthly.groupby('year')['date'].nunique()

# %% [markdown]
# ## Prepare Data for Training

# %%
train_val_test_df = pd.get_dummies(agg_df_with_monthly[agg_df_with_monthly['year'].isin([2013, 2014, 2015])],
                                   columns=['event_name_1', 'event_type_1',
                                            'event_name_2', 'event_type_2',
                                            'wday', 'month'],
                                   drop_first=True)

# convert boolean columns to int
for col in train_val_test_df.columns:
    if train_val_test_df[col].dtype == 'bool':
        train_val_test_df[col] = train_val_test_df[col].astype(int)


train_val_test_df

# %%
train_val_test_df.groupby('year')['date'].size()

# %%
train_val_test_df.columns

# %%
train_val_test_df['date'] = pd.to_datetime(train_val_test_df['date'])

train_val_test_df['day_of_month'] = train_val_test_df['date'].dt.day
train_val_test_df['day_of_month_norm'] = (train_val_test_df['day_of_month'] - 1) / 30

train_val_test_df['days_to_last_day'] = train_val_test_df['date'].dt.days_in_month - train_val_test_df['day_of_month']
train_val_test_df['days_to_last_day_norm'] = train_val_test_df['days_to_last_day'] / train_val_test_df['date'].dt.days_in_month

train_val_test_df.drop(columns=['day_of_month', 'days_to_last_day'], inplace=True)

# %%
train_val_test_df.columns

# %%
train_df = train_val_test_df[train_val_test_df['year'] == 2013].copy()
val_df = train_val_test_df[train_val_test_df['year'] == 2014].copy()
test_df = train_val_test_df[train_val_test_df['year'] == 2015].copy()

# %%
# Create all pairwise interaction terms.
# Since we are using simple models, we will manually create the interaction
# terms to allow the models to learn from them.
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

X_train = poly.fit_transform(train_df.drop(columns=['date', 'year', 'monthly_gmv_share']))
y_train = train_df['monthly_gmv_share']

X_val = poly.fit_transform(val_df.drop(columns=['date', 'year', 'monthly_gmv_share']))
y_val = val_df['monthly_gmv_share']

X_test = poly.fit_transform(test_df.drop(columns=['date', 'year', 'monthly_gmv_share']))
y_test = test_df['monthly_gmv_share']

# %% [markdown]
# ## Training

# %%
random_state = 42


# %%
def run_study(objective, n_trials=100, direction='minimize'):
    started_at = datetime.now()

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    ended_at = datetime.now()

    duration_in_seconds = (ended_at - started_at).total_seconds()

    print("Best parameters:", study.best_params)
    print("Best score:", study.best_value)
    print("Time taken:", humanize.naturaldelta(timedelta(seconds=duration_in_seconds)))

    return study, duration_in_seconds

def trial_evaluation_metric(y_true, y_pred):
    return metrics.mean_absolute_error(y_true, y_pred)

# %%
def objective_ridge(trial):
    params = {
        'alpha': trial.suggest_float("alpha", 1e-4, 100.0, log=True),  # Regularization strength
        'fit_intercept': trial.suggest_categorical("fit_intercept", [True, False]),
        'solver': trial.suggest_categorical("solver", ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
    }

    model = Ridge(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return trial_evaluation_metric(y_val, y_pred)


# %%
def objective_elastic_net(trial):
    params = {
        'alpha': trial.suggest_float("alpha", 1e-4, 1.0, log=True),
        'l1_ratio': trial.suggest_float("l1_ratio", 0.0, 1.0),
        'fit_intercept': trial.suggest_categorical("fit_intercept", [True, False]),
        'selection': trial.suggest_categorical("selection", ['cyclic', 'random'])
    }

    model = ElasticNet(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return trial_evaluation_metric(y_val, y_pred)


# %%
def objective_dt(trial):
    params = {
        'criterion': trial.suggest_categorical("criterion", ["squared_error", "friedman_mse", "absolute_error", "poisson"]),
        'max_depth': trial.suggest_int("max_depth", 1, 15),
        'min_samples_split': trial.suggest_int("min_samples_split", 5, 20),
        'min_samples_leaf': trial.suggest_int("min_samples_leaf", 5, 20),
        'max_features': trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }

    model = DecisionTreeRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return trial_evaluation_metric(y_val, y_pred)


# %%
def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int("n_estimators", 100, 1000),
        'max_depth': trial.suggest_int("max_depth", 2, 30),
        'min_samples_split': trial.suggest_int("min_samples_split", 2, 20),
        'min_samples_leaf': trial.suggest_int("min_samples_leaf", 2, 20),
        'max_features': trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        'bootstrap': trial.suggest_categorical("bootstrap", [True, False]),
    }

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return trial_evaluation_metric(y_val, y_pred)


# %%
def objective_xgb(trial):
    params = {
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'n_estimators': trial.suggest_int("n_estimators", 100, 1000),
        'max_depth': trial.suggest_int("max_depth", 3, 20),
        'learning_rate': trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        'subsample': trial.suggest_float("subsample", 0.5, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'gamma': trial.suggest_float("gamma", 0, 5),
        'reg_alpha': trial.suggest_float("reg_alpha", 0, 5),
        'reg_lambda': trial.suggest_float("reg_lambda", 0, 5),
    }


    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return trial_evaluation_metric(y_val, y_pred)

# %%
studies = [
    ('Ridge', objective_ridge, Ridge),
    ('Elastic Net', objective_elastic_net, ElasticNet),
    ('Decision Tree', objective_dt, DecisionTreeRegressor),
    ('Random Forest', objective_rf, RandomForestRegressor),
    ('XGBoost', objective_xgb, XGBRegressor),
        ]

# %%
training_results = []

for name, objective, model_class in studies:
    print(f"Running study for {name}...")
    study, execution_seconds = run_study(objective, n_trials=50)
    training_results.append(
            (name, model_class, study.best_params, execution_seconds)
            )

training_results_df = pd.DataFrame(
    training_results,
    columns=['Model', 'Model Class', 'Best Parameters', 'Execution Seconds']
    )

# %%
models = []
# fit best model
for name, model_class, best_params, execution_seconds in training_results:
    print(f"Fitting best model for {name} with parameters {best_params}...")
    best_model = model_class(**best_params).fit(X_train, y_train)

    models.append((name, best_model))

print()

# %%
for name, model in models:
    joblib.dump(model, f'models/Best {name}.joblib')


# %%
def get_mae(model, X, y, pred_df):
    """Calculate Mean Absolute Error for a given model."""
    __pred_df = pred_df.copy()

    y_pred = model.predict(X)
    __pred_df[f'pred'] = y_pred
    __pred_df[f'pred'] = __pred_df[f'pred'].clip(lower=0)
    __pred_df[f'pred_norm'] = __pred_df.groupby('yearmonth')[f'pred'].transform(lambda x: x / x.sum())
    mae = trial_evaluation_metric(y, __pred_df[f'pred'])
    mae_norm = trial_evaluation_metric(y, __pred_df[f'pred_norm'])

    return (
            mae,
            mae_norm,
            __pred_df[f'pred_norm'],
            )


# %% [markdown]
# ## Evaluate Models

# %%
results_df = pd.DataFrame()

pred_df = test_df.copy()
pred_df['date'] = pd.to_datetime(pred_df['date'])
pred_df['yearmonth'] = pred_df['date'].dt.to_period('M').astype(str)

for name, model in models:
    print(f"Predicting with {name}...")

    mae_train, normalized_mae_train, pred_norm = get_mae(model, X_train, y_train, pred_df)
    pred_df[f'{name}_train_pred_norm'] = pred_norm

    mae_val, normalized_mae_val, pred_norm = get_mae(model, X_val, y_val, pred_df)
    pred_df[f'{name}_val_pred_norm'] = pred_norm

    results_df = pd.concat([
        results_df,
        pd.DataFrame({
            'Model': [name],
            'MAE Train': [mae_train],
            'MAE Val': [mae_val],
            'Normalized MAE Train': [normalized_mae_train],
            'Normalized MAE Val': [normalized_mae_val],
        })
        ], ignore_index=True)


results_df.sort_values(by='MAE Val', ascending=True, inplace=True)

results_df.round(6)

# %%
pred_df.head(3).T

# %%

# %%
best_model_name = results_df.iloc[0]['Model']
mean_y_val = y_val.to_frame().mean().iloc[0]
median_y_val = y_val.to_frame().median().iloc[0]
min_mae = results_df['Normalized MAE Val'].min()

print(f'Best model: {best_model_name}')
print(f"MAE as a percentage of the mean:   {min_mae / mean_y_val * 100:.2f}%")
print(f"MAE as a percentage of the median: {min_mae / median_y_val * 100:.2f}%")

# %% [markdown]
# Having selected our champion model, let's see how it fares on the test set.

# %%
mae_test, normalized_mae_test, pred_norm = get_mae(
    models[0][1], X_test, y_test, pred_df
)

pred_df[f'{best_model_name}_test_pred_norm'] = pred_norm

mean_y_test = y_test.to_frame().mean().iloc[0]
median_y_test = y_test.to_frame().median().iloc[0]

print(f"{best_model_name} Normalized MAE Test: {normalized_mae_test:.6f}")
print(f"MAE Test as a percentage of the mean:   {normalized_mae_test / mean_y_test * 100:.2f}%")
print(f"MAE Test as a percentage of the median: {normalized_mae_test / median_y_test * 100:.2f}%")

# %%
# print execution times
for name, _, _, execution_seconds in training_results:
    humanized_execution_seconds = humanize.naturaldelta(
            timedelta(seconds=execution_seconds)
            )

    print(f"{name} execution time: {humanized_execution_seconds}")

total_execution_seconds = sum(
        execution_seconds for _, _, _, execution_seconds in training_results
        )

humanized_execution_seconds = humanize.naturaldelta(
        timedelta(seconds=total_execution_seconds)
        )

print()
print(f"Total execution time: {humanized_execution_seconds}")

# %%
with open(f'models/Best {best_model_name}.joblib', 'rb') as f:
    best_model = joblib.load(f)

best_model

# %%

# %%
y_test.to_frame().describe()

# %% [markdown]
# ## Sanity checks
# Make sure the normalized predictions sum up to 1 for each month.

# %%
pred_df.groupby('yearmonth')[f'{best_model_name}_train_pred_norm'].sum().round(4).value_counts()

# %%
model_name = best_model_name

for set_name in ['train', 'val', 'test']:
    value_counts = pred_df.groupby('yearmonth')[f'{model_name}_{set_name}_pred_norm'].sum().round(4).value_counts()

    assert list(value_counts.index) == [1], "Index is not [1]. Make sure the predictions are normalized correctly."
    assert list(value_counts.values) == [12], "Values are not [12]. Make sure the predictions are normalized correctly."

print('All sanity checks passed. Predictions are normalized correctly.')

# %% [markdown]
# ## Visualize the results
# We are going to compare the actual GMV with the predicted GMV on a daily
# basis to see how well the model performs. If the model accurately predicts the
# GMV using the monthly GMV share, it will be accurate enough to adjust the
# monthly targets to a daily level and later compare them with the actual
# GMV to evaluate our progress towards the monthly target.

# %%
viz_df = agg_df[['date', 'month', 'year', 'gmv']].merge(
        monthly_df[['year', 'month', 'gmv']],
        on=['year', 'month'],
        how='left',
        suffixes=('', '_monthly')
        ).rename(columns={'gmv': 'daily_gmv', 'gmv_monthly': 'monthly_gmv'})

viz_df['date'] = pd.to_datetime(viz_df['date'])

viz_df = viz_df.merge(
    pred_df[['date', f'{model_name}_test_pred_norm']].rename(columns={f'{model_name}_test_pred_norm': 'predicted_gmv_share'}),
    on='date',
    how='right'
        )

viz_df['predicted_daily_gmv'] = (viz_df['predicted_gmv_share'] * viz_df['monthly_gmv']).round(2)

viz_df['daily_gmv_cumulative'] = viz_df.groupby(['year', 'month'])['daily_gmv'].cumsum()
viz_df['predicted_daily_gmv_cumulative'] = viz_df.groupby(['year', 'month'])['predicted_daily_gmv'].cumsum()

viz_df

# %% [markdown]
# ### Actual vs Predicted Daily GMV

# %%
fig, ax = plt.subplots(
    figsize=(10, 5),
    dpi=440
)

actual_gmv_color = '#1f77b4'  # Default seaborn blue
predicted_gmv_color = '#ff7f0e'  # Default seaborn orange

ax.set_ylim(0, viz_df['daily_gmv'].max() * 1.1)

sns.lineplot(
    data=viz_df,
    x='date',
    y='daily_gmv',
    color=actual_gmv_color,
    ax=ax,
)

sns.lineplot(
    data=viz_df,
    x='date',
    y='predicted_daily_gmv',
    color=predicted_gmv_color,
    ax=ax,
)

# Title
text_properties = {
    'x': 0,
    'y': 0,
    'ha': 'left',
    'va': 'bottom',
    'fontsize': 14,
        }

t1 = ax.text(
        s='Actual',
        color=actual_gmv_color,
        **text_properties
        )

t2 = ax.text(
        s='vs',
        **text_properties
        )

t3 = ax.text(
        s='Predicted',
        color=predicted_gmv_color,
        **text_properties
        )

t4 = ax.text(
        s='Daily GMV',
        **text_properties
        )

dummy_title = ax.set_title('Daily GMV vs Predicted Daily GMV', fontsize=14)

title_coords = get_text_coordinates(dummy_title, ax, fig)
dummy_title.set_visible(False)

x = title_coords['x0']
y = title_coords['y0']

t1.set_position((x, y))
x = get_text_coordinates(t1, ax, fig)['x1']
t2.set_position((x, y))
x = get_text_coordinates(t2, ax, fig)['x1'] + 3  # small offset
t3.set_position((x, y))
x = get_text_coordinates(t3, ax, fig)['x1']
t4.set_position((x, y))

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('GMV', fontsize=12)

# Set tick font size
ax.tick_params(axis='both', which='major', labelsize=10)

# hide legend
ax.legend_.remove()

# format the y axis to have 
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{int(y / 1000)}k'))

# plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%

# %% [markdown]
# ### Actual vs Predicted Cumulative Daily GMV

# %%
fig, ax = plt.subplots(
    figsize=(10, 5),
    dpi=440
)

actual_gmv_color = '#1f77b4'  # Default seaborn blue
predicted_gmv_color = '#ff7f0e'  # Default seaborn orange

ax.set_ylim(0, viz_df['daily_gmv_cumulative'].max() * 1.1)

sns.lineplot(
    data=viz_df,
    x='date',
    y='daily_gmv_cumulative',
    color=actual_gmv_color,
    ax=ax,
)

sns.lineplot(
    data=viz_df,
    x='date',
    y='predicted_daily_gmv_cumulative',
    color=predicted_gmv_color,
    ax=ax,
)

# Title
text_properties = {
    'x': 0,
    'y': 0,
    'ha': 'left',
    'va': 'bottom',
    'fontsize': 14,
        }

t1 = ax.text(
        s='Actual',
        color=actual_gmv_color,
        **text_properties
        )

t2 = ax.text(
        s='vs',
        **text_properties
        )

t3 = ax.text(
        s='Predicted',
        color=predicted_gmv_color,
        **text_properties
        )

t4 = ax.text(
        s='Daily Cumulative GMV',
        **text_properties
        )

dummy_title = ax.set_title('Daily GMV vs Predicted Daily Cumulative GMV', fontsize=14)

title_coords = get_text_coordinates(dummy_title, ax, fig)
dummy_title.set_visible(False)

x = title_coords['x0']
y = title_coords['y0']

t1.set_position((x, y))
x = get_text_coordinates(t1, ax, fig)['x1']
t2.set_position((x, y))
x = get_text_coordinates(t2, ax, fig)['x1'] + 3  # small offset
t3.set_position((x, y))
x = get_text_coordinates(t3, ax, fig)['x1']
t4.set_position((x, y))

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('GMV', fontsize=12)

# Set tick font size
ax.tick_params(axis='both', which='major', labelsize=10)

# hide legend
# ax.legend_.remove()

# format the y axis to have 
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{int(y / 1000)}k'))

# plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%

# %% [markdown]
# ## Conclusion
#
# The models appears quite accurate. It is mostly having trouble with large spikes and dips, but as we see from the visualization of the cumulative gmv, it evens out so it shouldn't be an issue.
#
# We can safely use the model to adjust our monthly targets to a daily level and compare them with the actual metric to evaluate our progress.

# %%
