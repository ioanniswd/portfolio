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
# Customer Clustering
# ===================
# Using PCA and DBSCAN to cluster customers based on their demographic and
# behavioral data.

# %%
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.decomposition import PCA
import optuna
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_ind
from datetime import datetime, timedelta
import humanize

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
# Source: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
original_df = pd.read_csv('data/marketing_campaign.csv', sep='\t')

# %%
df = original_df.copy()

# %%
df

# %%
df.info()

# %%
df.isna().sum().sum()

# %%
na = df.isna().sum()
na[na > 0]

# %% [markdown]
# Just a few NA values in the income column. We can fill them with the median.
# or drop them entirely.

# %%
for col in df.select_dtypes(include='object'):
    print(f'{col}: {df[col].nunique()}')

# %% [markdown]
# From the datasets documentation we know that `Dt_Customer` is the date when
# the customer was registered. Let's convert it to a datetime object.

# %%
df['Dt_Customer'].head(10)

# %% [markdown]
# The format is day-month-year.

# %%
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')

# %%
for col in df.select_dtypes(include='object'):
    print(df[col].value_counts())
    print()

# %% [markdown]
# We could convert the `Education` column to an ordinal column.

# %%
df['EducationOrdinal'] = df['Education'].map({
    'Basic': 0,
    '2n Cycle': 1,
    'Graduation': 2,
    'Master': 3,
    'PhD': 4
    })

df.drop(columns=['Education'], inplace=True)

# %%
pd.DataFrame(df['EducationOrdinal'].value_counts().sort_index()).style.background_gradient()

# %%
df.drop(columns=['ID', 'Marital_Status'], inplace=True)

# %%
df.head(3).T

# %%
# heatmap
plt.figure(figsize=(18, 10))

sns.heatmap(
        df.corr(),
        annot=True,
        fmt='.1f',
        cmap='Oranges'
        )

plt.grid(False)

plt.title('Correlation Matrix', fontsize=16, fontweight=bold)
plt.show()

# %% [markdown]
# What about those columns with no correlation `Z_CostContact` and `Z_Revenue`.

# %%
df[['Z_CostContact', 'Z_Revenue']].head()

# %%
df[['Z_CostContact', 'Z_Revenue']].nunique()

# %% [markdown]
# Same value for all rows. We can drop them.

# %%
nunique_values = df.nunique()
nunique_values[nunique_values == 1]

# %%
df.drop(columns=['Z_CostContact', 'Z_Revenue'], inplace=True)

# %%
df.nunique().sort_values()

# %% [markdown]
# From the dataset documentation:
# - `Kidhome`: Number of children in customer's household
# - `Teenhome`: Number of teenagers in customer's household
# - `AcceptedCmpX`: 1 if customer accepted the offer in the Xst campaign, 0 otherwise
# - `Complain`: 1 if customer complained in the last 2 years, 0 otherwise
# - `Response`: 1 if customer accepted the offer in the last campaign, 0 otherwise

# %%
df['Kidhome'].value_counts()

# %%
df['Teenhome'].value_counts()

# %% [markdown]
# Let's drop the categorical columns for now.

# %%
df.drop(columns=['AcceptedCmp1', 'AcceptedCmp2',
                 'AcceptedCmp3', 'AcceptedCmp4',
                 'AcceptedCmp5', 'Complain', 'Response'
                 ], inplace=True)

# %%
# non-numerical columns
df.select_dtypes(exclude='number').columns

# %% [markdown]
# Let's convert that to days since registration.

# %%
df['DaysSinceRegistration'] = (pd.Timestamp.now() - df['Dt_Customer']).dt.days
df['DaysSinceRegistration'].head()

# %%
df['DaysSinceRegistration'].describe()

# %%
df.drop(columns=['Dt_Customer'], inplace=True)

# %%
# Find columns with high correlation
corr = df.corr().abs()
np.fill_diagonal(corr.values, np.nan)
high_corr = (corr >= 0.6).any()
high_corr_cols = high_corr[high_corr].index
high_corr_cols

# %%
# Compute correlation matrix
corr_matrix = df.corr()

# Filter columns with at least one correlation >= 0.6 (excluding diagonal)
mask = (corr_matrix.abs() > 0.6) & (corr_matrix.abs() < 1.0)
filtered_columns = corr_matrix.columns[mask.any(axis=1)]

# Filter the correlation matrix
filtered_corr = corr_matrix.loc[filtered_columns, filtered_columns]

plt.figure(figsize=(10, 6))

sns.heatmap(
        filtered_corr,
        annot=True,
        fmt='.2f',
        cmap='Oranges'
        )

plt.grid(False)

plt.title('High Correlation Matrix', fontsize=16, fontweight=bold)
plt.show()


# %% [markdown]
# Some correlations here, nothing too high. We can keep all columns for now.

# %% [markdown]
# ## Outliers

# %%
def iqr_bounds(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return lower_bound, upper_bound


# %%
outliers_df = pd.DataFrame()

for column in df.select_dtypes(include='number'):
    lower_bound, upper_bound = iqr_bounds(df[column])

    percentage_outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).mean()

    outliers_df = pd.concat([
        outliers_df,
        pd.DataFrame([[column, percentage_outliers]], columns=['Column', 'Percentage of Outliers'])
    ])

outliers_df.sort_values(by='Percentage of Outliers', ascending=False, inplace=True)
outliers_df['Outliers %'] = (outliers_df['Percentage of Outliers'] * 100).round(2).astype(str) + '%'
outliers_df[['Column', 'Outliers %']]

# %% [markdown]
# ## Preprocessing

# %%
median_income = df['Income'].median()
df['Income'].fillna(median_income, inplace=True)
median_income

# %%
# Based on the above, created the following preprocessor
from preprocessing.preprocessor import Preprocessor

# %% [markdown]
# #### PCA

# %%
df.isna().sum()

# %%
# Number of features
len(df.columns)

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

pca = PCA(n_components=0.95, random_state=random_state)
X_pca = pca.fit_transform(X_scaled)

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
# ## Training
# With optuna for hyperparameter tuning

# %%
preprocessing_pipeline = Pipeline([
    ('preprocessor', Preprocessor()),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95, random_state=random_state))
])


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

    return study


# %%
# Keeping 10 rows to simulate unseen data points being assigned to clusters
# using the persisted pipeline
seen_data = original_df[:-10].copy()
new_data = original_df[-10:].copy()

X = preprocessing_pipeline.fit_transform(seen_data)


# %%
def find_optimal_clusters(X, max_k=10):
    inertia_values = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        inertia_values.append(kmeans.inertia_)

    # Find the elbow point
    elbow = KneeLocator(range(2, max_k + 1), inertia_values, curve="convex", direction="decreasing").elbow
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_k + 1), inertia_values, marker="o")
    plt.axvline(elbow, color="r", linestyle="--", label=f"Optimal K: {elbow}")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.legend()
    plt.title("Elbow Method for Optimal K")
    plt.show()

    return elbow

optimal_k = find_optimal_clusters(X, max_k=10)


# %%
def objective(trial):
    n_clusters = trial.suggest_int(
            "n_clusters",
            max(2, optimal_k - 2),
            min(10, optimal_k + 2)
            )

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)

    num_clusters = len(set(labels))
    if num_clusters < 2:
        return float("-inf")

    return silhouette_score(X, labels)

study = run_study(objective, n_trials=100)

# %%
print('Best parameters:', study.best_params)
print('Best silhouette score:', study.best_value)

# %% [markdown]
# The best fit according to silhouette score appears for 3 clusters.

# %%
pipeline = Pipeline([
    ('preprocessing', preprocessing_pipeline),
    ('kmeans', KMeans(n_clusters=study.best_params['n_clusters'], random_state=random_state))
])

# %%
pipeline

# %%
pipeline.fit(seen_data)
seen_data['Cluster'] = pipeline.predict(seen_data)
seen_data['Cluster'].value_counts()

# %% [markdown]
# ## Store the fitted pipeline
# For future assignments of clusters

# %%
model_dir = Path('models')
# Creates folder if it doesn’t exist
model_dir.mkdir(exist_ok=True)

# %%
model_filename = model_dir / 'customer_clustering_pipeline.joblib'
model_filename

# %%
joblib.dump(pipeline, model_filename)

# %% [markdown]
# ## Use the persisted pipeline
# To assign clusters to new data points

# %%
pipeline = joblib.load(model_filename)

# %%
new_data['Cluster'] = pipeline.predict(new_data.copy())
new_data['Cluster'].value_counts()

# %% [markdown]
# ## Interpretation
# Using decision trees to interpret the clusters

# %%
preprocessed_data = Pipeline([
    ('preprocessor', Preprocessor()),
    ('scaler', StandardScaler())
]).fit_transform(seen_data.drop(columns=['Cluster']))

X_train, X_test, y_train, y_test = train_test_split(
    preprocessed_data,
    seen_data['Cluster'],
    test_size=0.2,
    random_state=random_state
)

rf = RandomForestClassifier(class_weight='balanced', random_state=random_state)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))

# %%
seen_data['Cluster'].value_counts()

# %% [markdown]
# Considering the above f1-score and the fact that all clusters are represented in the dataset, we can conclude that the random forest does a great job at identifying clusters based on the features provided.
# <br />
# Let's look at the feature importance.

# %%
feature_importance = rf.feature_importances_
features = Preprocessor().fit_transform(seen_data.drop(columns=['Cluster'])).columns

# %%
viz_df = pd.DataFrame({
          'Features': features,
          'Feature Importance': feature_importance / feature_importance.sum()
        }).sort_values(by='Feature Importance', ascending=False)

viz_df['Feature Importance Cumulative'] = viz_df['Feature Importance'].cumsum() * 100
viz_df.drop(columns=['Feature Importance'], inplace=True)

pareto(
        viz_df=viz_df,
        component_column='Features',
        cumulative_column='Feature Importance Cumulative',
        bold_key=bold
        )

# %%
viz_df

# %%
viz_df['non_cumulative'] = viz_df['Feature Importance Cumulative'] - viz_df['Feature Importance Cumulative'].shift(1).fillna(0)
important_features = viz_df[viz_df['non_cumulative'] > 5]['Features'].tolist()
important_features

# %% [markdown]
# The features that stand out accounting for a little over 80% of the feature importance are:
# 1. 'MntWines'
# 2. 'MntMeatProducts'
# 3. 'Income'
# 4. 'Teenhome'
# 5. 'NumCatalogPurchases'
# 6. 'NumDealsPurchases'
# 7. 'NumStorePurchases'
# 8. 'MntFishProducts'
# 9. 'MntSweetProducts'

# %% [markdown]
# How different is the amount spent on meat for each cluster?

# %%
for feature in important_features:
    plt.figure(figsize=(12, 6))

    sns.boxplot(
            x='Cluster',
            y=feature,
            data=seen_data
            )

    plt.title(feature,
              fontsize=16,
              fontweight=bold
              )
    plt.show()

# %% [markdown]
# The number of teenagers in the household(Teenhome) appear to be present almost exclusively in cluster 2

# %%
seen_data.groupby(['Cluster', 'Teenhome']).size()

# %% [markdown]
# ## Hypothesis Testing
#
# For each of the important features, we will test the following hypothesis:
# <br />
# $H_0$: The means of the clusters are **not** different.
# <br />
# $H_1$: The means of the clusters **are** different.

# %%
ALPHA = 0.05


# %%
def compare_clusters(df, column):
    cluster_0 = df[df['Cluster'] == 0][column]
    cluster_1 = df[df['Cluster'] == 1][column]
    cluster_2 = df[df['Cluster'] == 2][column]

    f_stat, p_value = f_oneway(cluster_0, cluster_1, cluster_2)

    if p_value < ALPHA:
        print('Reject the null hypothesis, the difference is statistically significant.')
        print()

        # Prepare data for Tukey's HSD test
        tukey_data = df[[column, 'Cluster']]
        tukey_data = tukey_data.dropna()
        tukey_data['Cluster'] = tukey_data['Cluster'].astype(str)  # Tukey test requires categorical groups

        # Perform Tukey's HSD test
        tukey_results = pairwise_tukeyhsd(endog=tukey_data[column],
                                          groups=tukey_data['Cluster'],
                                          alpha=ALPHA)
        print(tukey_results)
    else:
        print('Fail to reject the null hypothesis, the difference is not statistically significant.')


# %%
compare_clusters(seen_data, 'MntWines')

# %%
seen_data.groupby('Cluster')['MntWines'].describe().T

# %% [markdown]
# `Cluster 2` has the highest spending on wines with `Cluster 0` following closely behind. `Cluster 1` has a very low spending on wines compared to the other clusters.

# %%
compare_clusters(seen_data, 'MntMeatProducts')

# %%
seen_data.groupby('Cluster')['MntMeatProducts'].describe().T

# %% [markdown]
# `Cluster 2` spends the most on meat products. `Cluster 0` has a moderate spending on meat products while `Cluster 1` has a very low spending compared to the other clusters.

# %%
compare_clusters(seen_data, 'Income')

# %% [markdown]
# Income does not appear to differ significantly across clusters

# %%
compare_clusters(seen_data, 'Teenhome')

# %%
seen_data.groupby(['Cluster', 'Teenhome']).size()

# %% [markdown]
# `Cluster 0` appear more like to have at least one teenager in the household, `Cluster 2` appears more like to have none, and `Cluster 1` appears balanced between having and not having teenagers in the house.
# <br />
# In a future iteration, we could create a `Parent` categorical variable and try a clustering that allows for categorical variables.

# %%
compare_clusters(seen_data, 'NumCatalogPurchases')

# %%
seen_data.groupby('Cluster')['NumCatalogPurchases'].describe().T

# %% [markdown]
# `Cluster 2` has the greatest number of catalog purchases, `Cluster 0` has a moderate number of catalog purchases and `Cluster 1` has a low or negligible number of catalog purchases.

# %%
compare_clusters(seen_data, 'NumDealsPurchases')

# %%
seen_data.groupby('Cluster')['NumDealsPurchases'].describe().T

# %% [markdown]
# Customers in `Cluster 0` appear more likely to purchase a deal when compared with customers from `Cluster 1` and `Cluster 2` 

# %%
compare_clusters(seen_data, 'NumStorePurchases')

# %%
seen_data.groupby('Cluster')['NumStorePurchases'].describe().T

# %% [markdown]
# `Cluster 1` appears to make less store purchases than `Cluster 0` and `Cluster
# 2`. `Cluster 2` appears to make slightly more store purchases than `Cluster
# 1`, but the difference is small.

# %% [markdown]
# What is the distribution of web purchases across clusters? Is perhaps `Cluster
# 1` more likely to purchase online?

# %%
compare_clusters(seen_data, 'NumWebPurchases')

# %% [markdown]
# Nope, they simply appear to purchase less.

# %%
compare_clusters(seen_data, 'MntFishProducts')

# %%
seen_data.groupby('Cluster')['MntFishProducts'].describe().T

# %% [markdown]
# `Cluster 2` spends the most on fish products, `Cluster 0` a moderate to low amount, while `Cluster 1` has a low spending.

# %%
compare_clusters(seen_data, 'MntSweetProducts')

# %%
seen_data.groupby('Cluster')['MntSweetProducts'].describe().T

# %% [markdown]
# `Cluster 2` has the highest spending on sweet products, `Cluster 0` has a moderate spending and `Cluster 1` has a low spending.

# %% [markdown]
# ## Profiling
#
# Cluster 0: Average spenders
# - Purchaser of wines
# - Moderate spending on meat and sweet products
# - Moderate number of catalog purchases
# - More likely to purchase a deal
# - Moderate to low spending on fish products
#
# Cluster 1: Low spenders
# - Low spending on wines, meat, fish and sweet products
# - Low number of purchases in general, be it catalog, store or web purchases.
# - Less likely to purchase a deal
#
# Cluster 2: High spenders
# - Purchaser of wines
# - High spender on meats, fish and sweet products
# - High number of catalog purchases
# - Less likely to purchase a deal
#
# <br />
#
# Note: Income, although it was used in the clustering and might appear
# different across clusters, was not found to be statistically significant
# across clusters.

# %% [markdown]
# ## Conclusion
#
# Having split the customer base into the above clusters, we can better tailor
# marketing campaigns to each cluster. For instance, we could offer discounts
# and value-for-money packs to `Average spenders`, or promote premium products
# to `High spenders`.

# %% [markdown]
# Hope this analysis was insightful. Thank you for reading!
