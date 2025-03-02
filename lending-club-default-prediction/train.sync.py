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

# %%
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn import metrics
import optuna
from datetime import datetime, timedelta
import humanize

# %%
pd.set_option('display.float_format', '{:.2f}'.format)
random_state = 42

# %%
# Set default font
import matplotlib.font_manager as fm

font_path = '/usr/share/fonts/noto_sans_mono/NotoSansMono_SemiCondensed-SemiBold.ttf'
font_prop = fm.FontProperties(fname=font_path)

mpl.rcParams['font.family'] = font_prop.get_name()
plt.rcParams["font.weight"] = 'semibold'

bold = 'extra bold'

sns.set(font=font_prop.get_name(), style='darkgrid')

# %% [markdown]
# ## Model

# %% [markdown]
# We are going to use the following train-test split:
# - 60% training set to train the model
# - 20% validation set to tune the hyperparameters
# - 20% test set to evaluate the model after tuning

# %%
train_df = pd.read_csv('data/train.csv')

# %%
X = train_df.drop('default', axis=1)
y = train_df['default']

X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state
        )

X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        stratify=y_train_val,
        random_state=random_state
        )


# %%
def run_study(objective, n_trials=50, direction='maximize'):
    started_at = datetime.now()

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    ended_at = datetime.now()

    duration_in_seconds = (ended_at - started_at).total_seconds()

    print("Best parameters:", study.best_params)
    print("Best score:", study.best_value)
    print("Time taken:", humanize.naturaldelta(timedelta(seconds=duration_in_seconds)))

    return study, duration_in_seconds


# %%
def trial_evaluation_metric(y_true, y_pred):
    # return metrics.fbeta_score(y_true, y_pred, beta=10)
    return metrics.recall_score(y_true, y_pred)


# %%
def objective_dt(trial):
    params = {
            'max_depth': trial.suggest_int("max_depth", 1, 20),
            'min_samples_split': trial.suggest_int("min_samples_split", 2, 20),
            'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 20),
            'criterion': trial.suggest_categorical("criterion", ["gini", "entropy"]),
            'random_state': random_state
            }

    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return trial_evaluation_metric(y_val, y_pred)


# %%
def objective_rf(trial):
    params = {
            'n_estimators': trial.suggest_int("n_estimators", 10, 300),
            'max_depth': trial.suggest_int("max_depth", 1, 30),
            'min_samples_split': trial.suggest_int("min_samples_split", 2, 20),
            'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 20),
            'max_features': trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            'criterion': trial.suggest_categorical("criterion", ["gini", "entropy"]),
            'random_state': random_state
            }


    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return trial_evaluation_metric(y_val, y_pred)


# %%
def objective_xgb(trial):
    params = {
            'n_estimators': trial.suggest_int("n_estimators", 50, 500),
            'max_depth': trial.suggest_int("max_depth", 3, 15),
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            'subsample': trial.suggest_float("subsample", 0.5, 1.0),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
            'gamma': trial.suggest_float("gamma", 0, 5),
            'reg_alpha': trial.suggest_float("reg_alpha", 0, 10),
            'reg_lambda': trial.suggest_float("reg_lambda", 0, 10),
            'random_state': random_state,
            }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return trial_evaluation_metric(y_val, y_pred)


# %%
dt_study, dt_execution_seconds = run_study(objective_dt, n_trials=5)

# %%
rf_study, rf_execution_seconds = run_study(objective_rf, n_trials=5)

# %%
xgb_study, xgb_execution_seconds = run_study(objective_xgb, n_trials=5)

# %%
# create a df with the results
df_results = pd.DataFrame({
    'model': ['Decision Tree', 'Random Forest', 'XGBoost'],
    'evaluation_metric': [
        dt_study.best_value,
        rf_study.best_value,
        xgb_study.best_value
        ],
    'execution_seconds': [
        dt_execution_seconds,
        rf_execution_seconds,
        xgb_execution_seconds
        ]
    })

# %%
df_results


# %%
# calculate accuracy, precision, recall and f1-score
def calculate_metrics(model, X, y):
    y_pred = model.predict(X)

    accuracy = metrics.accuracy_score(y, y_pred)
    precision = metrics.precision_score(y, y_pred)
    recall = metrics.recall_score(y, y_pred)
    f1 = metrics.f1_score(y, y_pred)

    return accuracy, precision, recall, f1


# %%
best_dt = DecisionTreeClassifier(**dt_study.best_params)
best_rf = RandomForestClassifier(**rf_study.best_params)
best_xgb = XGBClassifier(**xgb_study.best_params)

best_dt.fit(X_train_val, y_train_val)
best_rf.fit(X_train_val, y_train_val)
best_xgb.fit(X_train_val, y_train_val)

''

# %%
# calculate metrics
dt_metrics = calculate_metrics(best_dt, X_test, y_test)
rf_metrics = calculate_metrics(best_rf, X_test, y_test)
xgb_metrics = calculate_metrics(best_xgb, X_test, y_test)

# %%
model_metrics = np.array([dt_metrics, rf_metrics, xgb_metrics]).transpose()
model_metrics

# %%
df_results['accuracy'] = model_metrics[0]
df_results['precision'] = model_metrics[1]
df_results['recall'] = model_metrics[2]
df_results['f1'] = model_metrics[3]

# %%
df_results

# %%
def roc_curve_metrics(model, X, y):
    y_pred_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred_proba)
    auc = metrics.roc_auc_score(y, y_pred_proba)

    return fpr, tpr, thresholds, auc

# %%
# plot auc with hue as model
fig, ax = plt.subplots(figsize=(10, 6))

viz_df = pd.DataFrame(columns=['model', 'fpr', 'tpr', 'auc', 'thresholds'])

fpr, tpr, thresholds, auc = roc_curve_metrics(best_dt, X_test, y_test)
viz_df = viz_df.append({
    'model': 'Decision Tree',
    'fpr': fpr,
    'tpr': tpr,
    'auc': auc,
    'thresholds': thresholds
    }, ignore_index=True)

fpr, tpr, thresholds, auc = roc_curve_metrics(best_rf, X_test, y_test)
viz_df = viz_df.append({
    'model': 'Random Forest',
    'fpr': fpr,
    'tpr': tpr,
    'auc': auc,
    'thresholds': thresholds
    }, ignore_index=True)

fpr, tpr, thresholds, auc = roc_curve_metrics(best_xgb, X_test, y_test)
viz_df = viz_df.append({
    'model': 'XGBoost',
    'fpr': fpr,
    'tpr': tpr,
    'auc': auc,
    'thresholds': thresholds
    }, ignore_index=True)

sns.lineplot(data=viz_df, x='fpr', y='tpr', hue='model', ax=ax)

plt.title('ROC Curve')

plt.show()
