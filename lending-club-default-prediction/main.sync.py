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
from tabulate import tabulate
import textwrap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

sns.set(font=font_prop.get_name())
mpl.rcParams['font.family'] = font_prop.get_name()
plt.rcParams["font.weight"] = 'semibold'

bold = 'extra bold'

sns.set_style(style='darkgrid')

# %%
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)

# %%
df_dict = pd.read_excel('data/LCDataDictionary.xlsx')
df_dict

# %%
df_dict.isna().sum()

# %%
df_dict.duplicated().sum()

# %%
df_dict.dropna(inplace=True)

# %%
df_dict.rename(columns={'LoanStatNew': 'column', 'Description': 'definition'}, inplace=True)

# %%
# split the text to lines of 80 characters
df_dict['definition'] = df_dict['definition'].apply(lambda x: '\n'.join(textwrap.wrap(x, width=80)))

# %%
df_dict['column'].str.startswith(' ').sum()

# %%
df_dict['column'].str.endswith(' ').sum()

# %%
df_dict['column'] = df_dict['column'].str.strip()

# %%
print(tabulate(df_dict, headers='keys', tablefmt='pipe'))

# %%
with open('data/data_dictionary.md', 'w') as f:
    f.write(tabulate(df_dict, headers='keys', tablefmt='pipe'))


# %%
def column_definition(col, df_dict=df_dict):
    """Get the definition of a column in the dataset."""
    try:
        return df_dict[df_dict['column'] == col]['definition'].values[0]
    except IndexError:
        return 'Definition not found'


# %%
column_definition('loan_amnt')

# %%
column_definition('column-not-found')

# %%
df0 = pd.read_csv('data/loan.csv', low_memory=False)
df0

# %%
set(df0.columns) - set(df_dict['column'])

# %%
set(df_dict['column']) - set(df0.columns)

# %% [markdown]
# Let's see if the missing `verification_status_joint` column is in fact the
# same as `verified_status_joint` in the data dictionary.

# %%
print(column_definition('verified_status_joint'))

# %%
df0['verification_status_joint'].value_counts(dropna=False)

# %% [markdown]
# Seems to be the same column. Let's rename it to `verification_status_joint`.

# %%
# change verified_status_joint to verification_status_joint
df_dict.loc[df_dict['column'] == 'verified_status_joint', 'column'] = 'verification_status_joint'

# %%
set(df0.columns) - set(df_dict['column'])

# %%
print(column_definition('verification_status_joint'))

# %% [markdown]
# Now we have definitions for all columns in the dataset.

# %%
nan_df = df0.isna().sum().to_frame('count')
nan_df['percentage'] = (nan_df['count'] / len(df0) * 100).map('{:.2f}%'.format)

# %%
print(nan_df[nan_df['count'] > 0].sort_values(['count'], ascending=False).to_string())

# %% [markdown]
# Columns id, url, and member_id have no data. Let's drop them.

# %%
df = df0.copy()
df.drop(columns=['id', 'url', 'member_id'], inplace=True)

# %%
# loan_status counts
viz_df = df['loan_status'].value_counts(dropna=False).to_frame('count')
viz_df['percentage'] = (viz_df['count'] / len(df) * 100)
viz_df['percentage_txt'] = viz_df['percentage'].map('{:.2f}%'.format)

fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(
    data=viz_df,
    x='percentage',
    y=viz_df.index
    )

max_percentage = viz_df['percentage'].max().astype(int) + 10

ax.set_xticklabels([f'{i}%' for i in range(0, max_percentage, 10)])

ax.set_xlim(0, max_percentage)

for i, row in viz_df.iterrows():
    ax.text(row['percentage'] + 0.5, i, row['percentage_txt'], va='center')

plt.show()

# %%
print(column_definition('loan_status'))

# %% [markdown]
# We could define a default as a loan in status `Default` or in status `Charged Off`. We will keep the defaults according to this definition, and the fully paid loans. 

# %%
df = df[df['loan_status'].isin(['Default', 'Charged Off', 'Fully Paid'])].copy()
df

# %%

# %%

# %%
print(column_definition('orig_projected_additional_accrued_interest'))

# %%
print(column_definition('deferral_term'))

# %%
print(column_definition('payment_plan_start_date'))

# %% [markdown]
# Most of those NaN are due to the user not having a hardship plan.

# %%
df['hardship_flag'].value_counts(dropna=False)

# %% [markdown]
# No nulls in the flag, let's use it.

# %%
non_hardship_nan_df = df[df['hardship_flag'] == 'Y'].isna().sum().to_frame('count')
non_hardship_nan_df['percentage'] = (non_hardship_nan_df['count'] / len(df) * 100).map('{:.2f}%'.format)
print(non_hardship_nan_df[non_hardship_nan_df['count'] > 0].sort_values(['count'], ascending=False).to_string())

# %% [markdown]
# The number of NaNs for the first few columns is the same as the number of users with `hardship flag == 'Y'`

# %%
print(column_definition('settlement_term'))

# %%
print(column_definition('settlement_status'))

# %%
print(column_definition('sec_app_mths_since_last_major_derog'))

# %%
print(column_definition('revol_bal_joint'))

# %%
df.head(10)

# %%
columns_to_keep = set(['loan_amnt'])

# %%
print(column_definition('funded_amnt'))

# %%
print(column_definition('funded_amnt_inv'))

# %%
columns_to_keep.add('term')

# %%
columns_to_keep.add('int_rate')

# %%
print(column_definition('installment'))

# %%
columns_to_keep.add('installment')

# %%
columns_to_keep.add('sub_grade')

# %%
print(column_definition('emp_title'))

# %% [markdown]
# Let's drop this column, it seems to be a text field provided by the user.

# %%
columns_to_keep.add('emp_length')

# %%
columns_to_keep = columns_to_keep.union(
        set(['home_ownership', 'annual_inc', 'verification_status', 'issue_d', 'loan_status'])
        )

# %% [markdown]
# Is `last_credit_pull_d` available at the time of inference (moment of application)?

# %%
df[['issue_d', 'last_credit_pull_d']]

# %% [markdown]
# Nope, it gets overwritten.

# %%
columns_to_keep

# %%
print(column_definition('pymnt_plan'))

# %%
df.groupby(['purpose', 'title']).size()

# %%
df['purpose'].value_counts()

# %%
df['title'].nunique()

# %%
columns_to_keep.add('purpose')

# %% [markdown]
# Let's just use the state, using the first 3 digits of the zip code seems a bit overkill for this first iteration.

# %%
columns_to_keep.add('addr_state')

# %%
(df['earliest_cr_line'].str.split('-').str[1].astype(int) > df['issue_d'].str.split('-').str[1].astype(int)).sum()

# %% [markdown]
# Field `earliest_cr_line` is not updated after the loan, it should be available at the time of inference.

# %%
columns_to_keep.add('earliest_cr_line')

# %%
print(column_definition('dti'))

# %%
columns_to_keep.add('dti')

# %%
print(column_definition('delinq_2yrs'))

# %% [markdown]
# Note sure if this is updated after the loan is issued.

# %%
print(column_definition('initial_list_status'))

# %%
columns_to_keep.add('initial_list_status')

# %%
print(column_definition('disbursement_method'))

# %%
columns_to_keep.add('disbursement_method')

# %%
columns_to_keep

# %%
cols = [col for col in df.columns if col in columns_to_keep]
df = df[cols].copy()

# %%
df['default'] = np.where(df['loan_status'] == 'Fully Paid', 0, 1)
df.drop(columns=['loan_status'], inplace=True)

# %%
df.head()

# %%
df['term'].value_counts()

# %%
# Keep only number of months
df['term'] = df['term'].str.replace('\D', '', regex=True).astype(int)

# %%
df.rename(columns={'term': 'term_in_months'}, inplace=True)

# %%
df['term_in_months'].value_counts()

# %%
df['sub_grade'].value_counts().sort_index().to_frame().style.background_gradient()


# %% [markdown]
# These look ordinal. Let's check.

# %%
def get_text_coordinates(text_element, ax, fig):
    """
    Get the coordinates of a text element on a given axis of a given figure.
    Used to position elements on the canvas.

    Parameters:
    text_element: Text element to get the coordinates of.
    ax: Axis of the figure.
    fig: Figure to get the coordinates of the text element on.

    Returns:
    Dictionary with the following keys:
    x0: coordinate of the text element on the x-axis.
    x1: coordinate of the text element on the x-axis.
    y0: coordinate of the text element on the y-axis.
    y1: coordinate of the text element on the y-axis.
    """
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


# %%
# create a stacked bar plot
viz_df = df.groupby(['sub_grade', 'default']).size().unstack().fillna(0).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))

zero_color = '#1f77b4'
one_color = '#ff7f0e'

viz_df.plot.barh(
    x='sub_grade',
    y=[0, 1],
    stacked=True,
    ax=ax,
    color=[zero_color, one_color],
    )

# hide legend
ax.get_legend().remove()

title_font_size = 18

# set title
title = ax.set_title(
        '_',
        fontsize=title_font_size,
        fontweight=bold
        )

coords = get_text_coordinates(title, ax, fig)
y0 = coords['y0']
title.set_visible(False)

x0 = 0

__text = ax.text(
        x=x0,
        y=y0,
        s='Paid',
        fontsize=title_font_size,
        fontweight=bold,
        color=zero_color,
        ha='left'
        )

x1 = get_text_coordinates(__text, ax, fig)['x1']

__text = ax.text(
        x=x1,
        y=y0,
        s=' vs ',
        fontsize=title_font_size,
        ha='left'
        )

x1 = get_text_coordinates(__text, ax, fig)['x1']

ax.text(
        x=x1,
        y=y0,
        s='Default',
        fontsize=title_font_size,
        fontweight=bold,
        color=one_color,
        ha='left'
        )

plt.show()

# %%
viz_df = df.groupby(['sub_grade', 'default']).size().unstack().fillna(0).reset_index()
viz_df['default_rate'] = viz_df[1] / (viz_df[0] + viz_df[1])

fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(
    data=viz_df,
    x='sub_grade',
    y='default_rate',
    color='blue'
    )

ax.set_title('Default rate by sub-grade', fontsize=18, fontweight=bold)

plt.show()

# %% [markdown]
# Definitely ordinal.

# %%
grades = df['sub_grade'].unique()
grades.sort()
grades

# %%
grade_map = {grade: i for i, grade in enumerate(grades)}
df['sub_grade_ordinal'] = df['sub_grade'].map(grade_map)

# %%
df[['sub_grade', 'sub_grade_ordinal']].head()

# %%
df.drop('sub_grade', axis=1, inplace=True)

# %%
df.head()

# %%
df['emp_length'].value_counts(dropna=False)

# %%
df['emp_length_in_years'] = df['emp_length'].replace({
    '< 1 year': '0 years',
    '10+ years': '10 years',
    '1 year': '1 years',
    }).str.replace('\D+', '', regex=True).astype(float)

df.drop('emp_length', axis=1, inplace=True)

# %%
df['emp_length_in_years'].value_counts(dropna=False)

# %%
df.head()

# %%
df['issued_month'] = df['issue_d'].str.split('-').str[0]
# To use in conjunction with the year of earliest credit line
df['issued_year'] = df['issue_d'].str.split('-').str[1].astype(int)

df.drop('issue_d', axis=1, inplace=True)

# %%
df['issued_month'].value_counts()

# %%
df['earliest_cr_line_month'] = df['earliest_cr_line'].str.split('-').str[0]
df['earliest_cr_line_year'] = df['earliest_cr_line'].str.split('-').str[1].astype(int)

df['cr_line_years_ago'] = df['issued_year'] - df['earliest_cr_line_year']

# %%
df['earliest_cr_line_month'].value_counts()

# %%
df['cr_line_years_ago'].describe()

# %%
df.drop([
          'issued_year',
          'earliest_cr_line',
          'earliest_cr_line_year'
        ],
        axis=1,
        inplace=True)

# %%
df.head()

# %%
len(df.columns)

# %%
df.info()

# %%
string_columns = [col for col in df.columns if df[col].dtype == 'object']
string_columns

# %%
df[string_columns].head()

# %%
df[list(set(df.columns) - set(string_columns))].head()

# %%
# one hot encode the string columns
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[string_columns])
encoded

# %%
df_encoded = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(string_columns)
        ).astype(int)

df_encoded.head()

# %%
train_df = df.drop(string_columns, axis=1).copy().reset_index(drop=True)
train_df = pd.concat([train_df, df_encoded], axis=1)

float_columns = train_df.select_dtypes(include='float').columns
train_df[float_columns] = train_df[float_columns].round(2) # to handle float precision

train_df

# %%
len(train_df.columns)

# %% [markdown]
# ## Model

# %% [markdown]
# We are going to use the following train-test split:
# - 60% training set to train the model
# - 20% validation set to tune the hyperparameters
# - 20% test set to evaluate the model after tuning

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
def objective_dt(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 5, 8),
        # 'max_depth': trial.suggest_int('max_depth', 2, 32),
        # 'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        # 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        # 'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
        'random_state': random_state
    }

    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return metrics.roc_auc_score(y_val, y_pred)


# %%
def objective_rf(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        # 'max_depth': trial.suggest_int('max_depth', 2, 32),
        # 'n_estimators': trial.suggest_int('n_estimators', 2, 200),
        # 'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        # 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        # 'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
        'random_state': random_state,
        'n_jobs': -1
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return metrics.roc_auc_score(y_val, y_pred)


# %%
def objective_xgb(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 5, 6),
        # 'max_depth': trial.suggest_int('max_depth', 2, 32),
        # 'n_estimators': trial.suggest_int('n_estimators', 2, 200),
        # 'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 1),
        # 'subsample': trial.suggest_discrete_uniform('subsample', 0.5, 1, 0.1),
        # 'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.1),
        # 'gamma': trial.suggest_loguniform('gamma', 0.001, 10),
        # 'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.001, 10),
        # 'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.001, 10),
        'random_state': random_state,
        'objective': 'binary:logistic',
        'n_jobs': -1
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return metrics.roc_auc_score(y_val, y_pred)


# %%
# %%time

dt_study, dt_execution_seconds = run_study(objective_dt, n_trials=3)
rf_study, rf_execution_seconds = run_study(objective_rf, n_trials=3)
xgb_study, xgb_execution_seconds = run_study(objective_xgb, n_trials=3)

# dt_study, dt_execution_seconds = run_study(objective_dt, n_trials=100)
# rf_study, rf_execution_seconds = run_study(objective_rf, n_trials=100)
# xgb_study, xgb_execution_seconds = run_study(objective_xgb, n_trials=100)

# %%
# create a df with the results
df_results = pd.DataFrame({
    'model': ['Decision Tree', 'Random Forest', 'XGBoost'],
    'roc_auc': [
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
dt_metrics

# %%
rf_metrics

# %%
xgb_metrics

# %%
df_results['accuracy'] = [dt_metrics[0], rf_metrics[0], xgb_metrics[0]]
df_results['precision'] = [dt_metrics[1], rf_metrics[1], xgb_metrics[1]]
df_results['recall'] = [dt_metrics[2], rf_metrics[2], xgb_metrics[2]]
df_results['f1'] = [dt_metrics[3], rf_metrics[3], xgb_metrics[3]]

# %%
df_results

# %%
