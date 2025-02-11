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
# https://www.kaggle.com/code/akankshajadhav24/wholesale-customers-data-using-pca-and-kmeans

# %%
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import DBSCAN
import gower
from sklearn.decomposition import PCA
import optuna
from sklearn.tree import DecisionTreeClassifier

import sys
sys.path.append('../lib')
from visualization_helpers import pareto

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
# Set default font
import matplotlib.font_manager as fm

font_path = '/usr/share/fonts/noto_sans_mono/NotoSansMono_SemiCondensed-SemiBold.ttf'
font_prop = fm.FontProperties(fname=font_path)

sns.set(font=font_prop.get_name())
mpl.rcParams['font.family'] = font_prop.get_name()
plt.rcParams["font.weight"] = 'semibold'

bold = 'extra bold'

# %%
random_state = 42

# %% [markdown]
# ## EDA

# %%
# Source: https://www.kaggle.com/datasets/binovi/wholesale-customers-data-set/
original_df = pd.read_csv('data/wholesale-customers-data-set.csv')

# %%
df = original_df.copy()

# %%
df

# %%
df.info()

# %%
df.isna().sum().sum()

# %%
df['Channel'].value_counts()

# %%
df['Region'].value_counts()

# %%
categorical_columns = ['Channel', 'Region']
numerical_columns = list(set(df.columns) - set(categorical_columns))

# sanity check
if set(numerical_columns + categorical_columns) != set(df.columns):
    raise ValueError("Numerical and categorical columns do not cover all columns")
else:
    print('Numerical columns:   ', len(numerical_columns))
    print('Categorical columns: ', len(categorical_columns))

# %%
# darkgrid
sns.set_theme(style="darkgrid")

# %%
sns.pairplot(df[numerical_columns], diag_kind='kde', height=1.5)

plt.show()

# %% [markdown]
# Notice that the distributions are right skewed. We can apply log
# transformation to make them more normal.

# %% [markdown]
# ## Preprocessing

# %%
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(df[numerical_columns].values)
X_scaled = np.concatenate([
    X_numerical_scaled,
    df[categorical_columns].values
    ], axis=1)

categorical_idx = [i for i in range(len(X_numerical_scaled[0]),
                                    len(X_scaled[0]))]


# %%
# df_log_scaled = df.copy()
# df_log_scaled[numerical_columns] = np.log1p(df_log_scaled[numerical_columns])
# X_numerical_log_scaled = df_log_scaled[numerical_columns].values
# X_log_scaled = np.concatenate([
#     X_numerical_log_scaled,
#     df_log_scaled[categorical_columns].values
#     ], axis=1)

# categorical_idx = [i for i in range(len(X_numerical_log_scaled[0]),
#                                     len(X_log_scaled[0]))]

# %%
# sns.pairplot(df_scaled[numerical_columns], diag_kind='kde', height=1.5)
# plt.show()

# %% [markdown]
# Check for multicollinearity, I noticed that `Detergents_Paper` and `Grocery` and `Milk` appear correlated.

# %%
# plt.figure(figsize=(8, 6))

# corr_matrix = df_log_scaled[numerical_columns].corr()
# sns.heatmap(corr_matrix, annot=True, cmap='Oranges')

# plt.title('Correlation Matrix')

# plt.show()

# %%
def run_study(objective, n_trials=50, direction='maximize'):
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Best parameters:", study.best_params)
    print("Best score:", study.best_value)

    return study

def percentage_value_counts(df, feature):
    return (df[feature].value_counts().astype(float) / len(df) * 100).map(lambda x: f"{x:.1f}%")

def mixed_distance(X, labels, categorical_indices):
    """Compute silhouette score for mixed-type data."""
    n_samples = X.shape[0]
    unique_clusters = np.unique(labels)
    silhouette_scores = np.zeros(n_samples)

    for i in range(n_samples):
        current_cluster = labels[i]
        same_cluster = X[labels == current_cluster]
        other_clusters = [X[labels == c] for c in unique_clusters if c != current_cluster]

        # Compute average intra-cluster distance (a(i))
        a_i = np.mean([
            np.linalg.norm(X[i, :].astype(float) - same_cluster[j, :].astype(float))
            for j in range(len(same_cluster)) if j != i
        ]) if len(same_cluster) > 1 else 0

        # Compute average nearest-cluster distance (b(i))
        b_i = np.min([
            np.mean([
                np.linalg.norm(X[i, :].astype(float) - cluster[j, :].astype(float))
                for j in range(len(cluster))
            ]) for cluster in other_clusters
        ]) if other_clusters else 0

        # Compute silhouette score for the point
        silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0

    return np.mean(silhouette_scores)


# %%
def objective_kprototypes_silhouette_scaled(trial):
    n_clusters = trial.suggest_int("n_clusters", 2, 10)
    gamma = trial.suggest_float("gamma", 0.1, 5.0)
    init = trial.suggest_categorical("init", ["Huang", "Cao"])

    kproto = KPrototypes(
            n_clusters=n_clusters,
            gamma=gamma,
            init=init,
            random_state=random_state
            )

    labels = kproto.fit_predict(
            X_scaled,
            categorical=categorical_idx
            )

    return mixed_distance(X_scaled, labels, categorical_idx)


# %%
study = run_study(objective_kprototypes_silhouette_scaled, n_trials=50)
scaled_best_score = study.best_value

# %%
kproto_scaled = KPrototypes(
        n_clusters=study.best_params['n_clusters'],
        gamma=study.best_params['gamma'],
        init=study.best_params['init'],
        random_state=random_state
        )

labels = kproto_scaled.fit_predict(
        X_scaled,
        categorical=categorical_idx
        )

pd.DataFrame(labels).value_counts()

# %% [markdown]
# Let's try without scaling

# %%
X = df.values
categorical_idx = [df.columns.get_loc(col) for col in categorical_columns]


# %%
def objective_kprototypes_silhouette_not_scaled(trial):
    n_clusters = trial.suggest_int("n_clusters", 2, 10)
    gamma = trial.suggest_float("gamma", 0.1, 5.0)
    init = trial.suggest_categorical("init", ["Huang", "Cao"])

    kproto = KPrototypes(
            n_clusters=n_clusters,
            gamma=gamma,
            init=init,
            random_state=random_state
            )

    labels = kproto.fit_predict(
            X,
            categorical=categorical_idx
            )

    return mixed_distance(X, labels, categorical_idx)


# %%
study = run_study(objective_kprototypes_silhouette_not_scaled, n_trials=50)
not_scaled_best_score = study.best_value

# %%
kproto_not_scaled = KPrototypes(
        n_clusters=study.best_params['n_clusters'],
        gamma=study.best_params['gamma'],
        init=study.best_params['init'],
        random_state=random_state
        )

labels = kproto_not_scaled.fit_predict(
        X,
        categorical=categorical_idx
        )

pd.DataFrame(labels).value_counts()

# %% [markdown]
# ## PCA

# %%
scaler = StandardScaler()

X_numerical_scaled = scaler.fit_transform(df[numerical_columns].values)
pca = PCA(n_components='mle')
X_pca = pca.fit_transform(X_numerical_scaled)
X_pca = np.concatenate([
    X_pca,
    df[categorical_columns].values
    ], axis=1)

# %%
viz_df = pd.DataFrame(
        np.cumsum(pca.explained_variance_ratio_),
        columns=['Explained Variance']
        )

viz_df['PC'] = [f'PC{x}' for x in range(1, len(viz_df) + 1)]
viz_df['Explained Variance'] = (viz_df['Explained Variance'] * 100).round(2)

pareto(
        viz_df=viz_df,
        component_column='PC',
        cumulative_column='Explained Variance',
        bold_key=bold
        )

# %% [markdown]
# Select the 4 components that explain > 90% of the variance.

# %%
n_components = 4
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_numerical_scaled)
X_pca = np.concatenate([
    X_pca,
    df[categorical_columns].values
    ], axis=1)


# %%
def objective_kprototypes_pca(trial):
    n_clusters = trial.suggest_int("n_clusters", 2, 10)
    gamma = trial.suggest_float("gamma", 0.1, 5.0)
    init = trial.suggest_categorical("init", ["Huang", "Cao"])

    kproto = KPrototypes(
            n_clusters=n_clusters,
            gamma=gamma,
            init=init,
            random_state=random_state
            )

    labels = kproto.fit_predict(
            X_pca,
            categorical=categorical_idx
            )

    return silhouette_score(X_pca, labels)


# %%
study = run_study(objective_kprototypes_pca, n_trials=50, direction='maximize')
pca_best_score = study.best_value

# %%
kproto_pca = KPrototypes(
        n_clusters=study.best_params['n_clusters'],
        gamma=study.best_params['gamma'],
        init=study.best_params['init'],
        random_state=random_state
        )

labels = kproto_pca.fit_predict(
        X,
        categorical=categorical_idx
        )

pd.DataFrame(labels).value_counts()

# %%
print('Scaled:', scaled_best_score)
print('Not scaled:', not_scaled_best_score)
print('PCA:', pca_best_score)

# %%
# Assume 'X' is your dataset and 'labels' are cluster assignments
clf = DecisionTreeClassifier()
clf.fit(X_scaled, kproto_scaled.predict(
        X_scaled,
        categorical=categorical_idx
        ))

# Feature importance
importances = clf.feature_importances_

# %%
series = pd.Series(
        importances,
        index=[*numerical_columns, *categorical_columns]
        ).sort_values(ascending=False)

__df = series.to_frame('Importance').reset_index(names='Feature')
__df

# %%
__df['Importance %'] = (__df['Importance'] * 100).round(2)
__df['Importance Cumulative %'] = np.cumsum(__df['Importance %'])

# %%
__df

# %%
pareto(
        viz_df=__df,
        component_column='Feature',
        cumulative_column='Importance Cumulative %',
        bold_key=bold
        )

# %%

# %%
# Assume 'X' is your dataset and 'labels' are cluster assignments
clf = DecisionTreeClassifier()
clf.fit(X, kproto_not_scaled.predict(
        X,
        categorical=categorical_idx
        ))

# Feature importance
importances = clf.feature_importances_

# %%
series = pd.Series(
        importances,
        index=[*numerical_columns, *categorical_columns]
        ).sort_values(ascending=False)

__df = series.to_frame('Importance').reset_index(names='Feature')
__df

# %%
__df['Importance %'] = (__df['Importance'] * 100).round(2)
__df['Importance Cumulative %'] = np.cumsum(__df['Importance %'])

# %%
__df

# %%
pareto(
        viz_df=__df,
        component_column='Feature',
        cumulative_column='Importance Cumulative %',
        bold_key=bold
        )

# %%

# %%
