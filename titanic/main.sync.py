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
# # Titanic: Machine Learning from Disaster

# %% [markdown]
# ## Imports and configurations

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression

import joblib
from pathlib import Path

# %%
# Set default font
import matplotlib.font_manager as fm

font_path = '/usr/share/fonts/noto_sans_mono/NotoSansMono_SemiCondensed-SemiBold.ttf'
font_prop = fm.FontProperties(fname=font_path)

mpl.rcParams['font.family'] = font_prop.get_name()
plt.rcParams["font.weight"] = 'semibold'

bold = 'extra bold'

sns.set(font=font_prop.get_name(), style='darkgrid')

# %%
random_state = 42

# %%
train_df = pd.read_csv('data/train.csv')
train_df

# %%
test_df = pd.read_csv('data/test.csv')
test_df.head(3)

# %%
train_df.isna().sum()

# %%
(train_df.isna().sum() / len(train_df) * 100).round(1).to_frame().style.background_gradient()

# %%
train_df['Sex'].value_counts()

# %%
train_df['Embarked'].value_counts(dropna=False)

# %%
train_df['Pclass'].value_counts()

# %%
train_df['Survived'].value_counts()

# %%
test_df.isna().sum()

# %%
(test_df.isna().sum() / len(test_df) * 100).round(1).to_frame().style.background_gradient()

# %% [markdown]
# We will drop the `Cabin` column.

# %%
train_df.drop('Cabin', axis=1, inplace=True)

# %%
cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

plt.figure(figsize=(10, 6))

sns.heatmap(
        train_df[cols].corr(),
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        vmin=-1,
        vmax=1
        )

plt.show()

# %% [markdown]
# Let's try to impute the missing values in the `Age` column. Going with a simple median imputation for now. We could look into something more sophisticated in a future iteration since there is some correlation between age and some of the other features.

# %%
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())

# %%
titles = train_df['Name'].str.extract(r',\s(.*?)\.')[0]
titles.value_counts(dropna=False)

# %%
train_df['Title'] = np.where(
        titles.isin(['Mr', 'Miss', 'Mrs', 'Master']),
        titles,
        'Other'
        )

# %%
train_df['Title'].value_counts(dropna=False)

# %%
train_df.drop(columns=['Name'], inplace=True)

# %%
train_df

# %% [markdown]
# Let's drop the ticket column, it doesn't seem to have information we can use.

# %%
train_df.drop('Ticket', axis=1, inplace=True)

# %% [markdown]
# Let's onehot encode the categorical columns to prepare for training.

# %%
train_df = pd.get_dummies(train_df,
                          columns=['Sex', 'Embarked', 'Title'],
                          drop_first=False).astype(float)

# %%
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1

# %%
train_df

# %% [markdown]
# Let's put all of this into a class so we can reuse it later.

# %%
from preprocessing.preprocessor import Preprocessor

# %%
preprocessed = Preprocessor().transform(pd.read_csv('data/train.csv'))
preprocessed

# %%
preprocessed.equals(train_df)

# %% [markdown]
# ## Model training

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn import metrics
import optuna
from datetime import datetime, timedelta
import humanize
from sklearn.tree import plot_tree

# %%
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.2,
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
#     return metrics.precision_score(y_true, y_pred)
    # return metrics.recall_score(y_true, y_pred)
    return metrics.f1_score(y_true, y_pred)
    # return metrics.fbeta_score(y_true, y_pred, beta=10)


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
dt_study, dt_execution_seconds = run_study(objective_dt, n_trials=50)

# %%
rf_study, rf_execution_seconds = run_study(objective_rf, n_trials=50)

# %%
xgb_study, xgb_execution_seconds = run_study(objective_xgb, n_trials=50)

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

best_dt.fit(X_train, y_train)
best_rf.fit(X_train, y_train)
best_xgb.fit(X_train, y_train)

print()

# %% [markdown]
# Save the models

# %%
joblib.dump(best_dt, 'models/best_dt.joblib')
joblib.dump(best_rf, 'models/best_rf.joblib')
joblib.dump(best_xgb, 'models/best_xgb.joblib')

# %%
# calculate metrics
dt_metrics = calculate_metrics(best_dt, X_val, y_val)
rf_metrics = calculate_metrics(best_rf, X_val, y_val)
xgb_metrics = calculate_metrics(best_xgb, X_val, y_val)

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


# %% [markdown]
# Similar performance for all models.

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

max_legend_length = max([len(model + ': ') for model in df_results['model']])

fpr, tpr, thresholds, auc = roc_curve_metrics(best_dt, X_val, y_val)
sns.lineplot(x=fpr, y=tpr, ax=ax, label=f'Decision Tree: '.ljust(max_legend_length) + f'(AUC={auc:.2f})')

fpr, tpr, thresholds, auc = roc_curve_metrics(best_rf, X_val, y_val)
sns.lineplot(x=fpr, y=tpr, ax=ax, label=f'Random Forest: '.ljust(max_legend_length) + f'(AUC={auc:.2f})')

fpr, tpr, thresholds, auc = roc_curve_metrics(best_xgb, X_val, y_val)
sns.lineplot(x=fpr, y=tpr, ax=ax, label=f'XGBoost: '.ljust(max_legend_length) + f'(AUC={auc:.2f})')

ax.set_xlabel('False Positive Rate', fontweight=bold)
ax.set_ylabel('True Positive Rate', fontweight=bold)

plt.title('ROC Curve', fontweight=bold, fontsize=16)

plt.show()

# %% [markdown]
# Similar AUC for all models, 
# <br />
# We can use the decision tree model to also take advantage of the interpretability.

# %%
dt_study.best_params

# %%
# print the tree

plt.figure(figsize=(20, 10))

plot_tree(
        best_dt,
        feature_names=X.columns,
        class_names=['Did not survive', 'Survived'],
        )

plt.show()


# %%
def color_confusion_matrix_annotations(cm, ax, i, j):
    # Get the colormap used in the plot
    cmap = ax.images[0].cmap
    norm = ax.images[0].norm

    # See https://github.com/scikit-learn/scikit-learn/blob/99bf3d8e4/sklearn/metrics/_plot/confusion_matrix.py#L159
    # print text with appropriate color depending on background
    thresh = (cm.max() + cm.min()) / 2.0

    cmap_max = cmap(norm(cm.max()))
    cmap_min = cmap(norm(cm.min()))

    color = cmap_max if cm[i, j] < thresh else cmap_min

    return color


# %%
def plot_confusion_matrix(model, X, y):
    y_pred = model.predict(X)

    cm = metrics.confusion_matrix(y, y_pred)
    disp = metrics.ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=X.columns,
            )

    fig, ax = plt.subplots(
            nrows=1, ncols=1,
            figsize=(4, 3),
            dpi=200
            )

    disp = metrics.ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Did not survive', 'Survived'],
            )

    disp.plot(ax=ax, values_format='.0f', cmap='Oranges')

    # remove grid lines
    ax.grid(False)

    annotations = [
        ["True Negative", "False Positive"],
        ["False Negative", "True Positive"],
    ]

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i - 0.2,  # Slightly above the center
                annotations[i][j],
                ha="center", va="center",
                fontsize=9,
                color=color_confusion_matrix_annotations(cm, ax, i, j)
            )

    # change tick font size
    ax.tick_params(axis='both', which='major', labelsize=9)

    report = metrics.classification_report(y, y_pred, output_dict=True)
    metrics_df = pd.DataFrame(report).T.round(3)

    ax.set_title(type(model).__name__, fontweight=bold, fontsize=13)

    ax.set_xlabel('Predicted label', fontweight=bold)
    ax.set_ylabel('True label', fontweight=bold)

    plt.show()


# %%
plot_confusion_matrix(best_dt, X_val, y_val)

# %%
plot_confusion_matrix(best_rf, X_val, y_val)

# %%
plot_confusion_matrix(best_xgb, X_val, y_val)

# %% [markdown]
# Similar performance for all models. We can use the decision tree model for its
# interpretability.

# %% [markdown]
# Predict on the test set

# %%
preprocessor = Preprocessor()
X_test = preprocessor.transform(test_df)
y_pred = best_dt.predict(X_test)

# %%
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': y_pred
    })

# %%
submission['Survived'].value_counts()

# %%
submission['Survived'] = submission['Survived'].astype(int)

# %%
submission.to_csv('data/submission.csv', index=False)

# %%
