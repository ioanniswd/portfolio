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
# ## Context
# My girlfriend's master's thesis was the `Perception of Anthropomorphic Traits in
# Cars`. She wanted to create a questionaire to test this hypothesis.
# <br />
# She had acquired a dataset of images of cars, and after selecting the images
# that could be used in the questionaire, she created a spreadsheet with the file name
# and the features for each car, such as the size of the grille, the shape of the
# headlights, etc.
# <br />
# She needed to select 10 images for the questionaire, and those images had to
# be representative of the different classes of the various labels, e.g. `Bumper
# Shape: upturned lower edge-straight upper edge` or `Headlights Position: only
# upper`.
#
#
# ## Approach
# I used [scikit-multilearn](http://scikit.ml/tutorial.html) to split the data into a
# training and a test set that were representative of the different classes of
# the various labels in the dataset, and exported the rows of the training set
# containing the filename, to a CSV file.
#
# Note: The data used in this notebook have been anonymized.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split

# %%
df = pd.read_csv('stimuli.csv')

# %%
df.columns

# %%
df

# %%
df.isna().sum()

# %% [markdown]
# No nulls

# %%
# Change order of columns based on stakeholder feedback
features = [
        'Car Size',
        'Grille Size',
        'Grille Shape',
        'Bumper Shape',
        'Headlights Position',
        'Headlights Shape',
        'Number of bulbs in each headlight',
        'Bulb Shape'
        ]


# %% [markdown]
# Percentage of rows, grouped by feature.

# %%
def percentage_value_counts(df, feature):
    return (df[feature].value_counts().astype(float) / len(df) * 100).map(lambda x: f"{x:.1f}%")


# %%
for feature in features:
    print(percentage_value_counts(df, feature))
    print("\n")

# %%
sns.set_style('darkgrid')


# %%
def plot_bar(df, column):
    fig, ax = plt.subplots()

    sns.barplot(
            df[column].value_counts(),
            ax=ax,
            orient='h'
            )

    # make xlabel bold
    ax.set_xlabel(ax.get_xlabel(), fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontweight='bold')
    ax.set_title(column, fontweight='bold', fontsize=15)

    plt.show()


# %%
for column in features:
    plot_bar(df, column)

# %% [markdown]
# Applying filters according to the charts above and according to stakeholder feedback to simplify the stimuli for this iteration.

# %%
normal_df = df[
    (df['Car Size'] == 'normal') &
    (df['Grille Size'] != 'does not have') &
    (df['Grille Shape'].isin(['lower edge upturned-upper edge straight', 'lower edge upturned-upper edge downturned'])) &
    (df['Bumper Shape'].isin(['upturned lower edge-straight upper edge', 'downturned upper edge-straight lower'])) &
    (df['Headlights Shape'].isin(['irregular', 'trapezoid-inner side downturned'])) &
    (df['Number of bulbs in each headlight'] == '2') &
    (df['Bulb Shape'].isin(['round', 'round and irregular']))
]

# %% [markdown]
# We are left with the following number of pictures

# %%
len(normal_df)

# %%
for feature in features:
    print(normal_df[feature].value_counts())
    print("\n")

# %%
normal_df

# %%
nunique = normal_df.nunique()
nunique

# %%
normal_df

# %%
df

# %% [markdown]
# Decided to drop bulb shape and headlights shape as features to reduce complexity.

# %%
to_drop = ['Filename', 'Bulb Shape', 'Headlights Shape']

# %%
final_features = [x for x in features if x not in to_drop]

# %%
dummy_df = pd.get_dummies(normal_df.drop(to_drop, axis=1), columns=final_features)

# %%
num_of_images = len(normal_df)
num_of_images

# %%
# Using a dummy X variable, we are only interested in the indices
# of the selected rows
X = np.zeros((num_of_images, 1))
y = dummy_df.to_numpy()

# %%
number_of_images_for_questionaire = 10

# %%
test_size = round(1 - number_of_images_for_questionaire * 1.0 / num_of_images, 2)
test_size

# %%
X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=test_size)

# %%
y_train

# %% [markdown]
# Let's find the rows of the train set and verify that all classes of all labels are represented. 

# %%
df.duplicated().sum()

# %% [markdown]
# No duplicates, we can use equality to find the indices of the selected rows.

# %%
indices = []

for y_values in y_train:
    for index, data in dummy_df.iterrows():
        if np.array_equal(data.to_numpy(), y_values):
            indices.append(index)
            break

# %%
indices

# %%
normal_df.loc[indices]

# %%
normal_df[final_features].nunique()

# %%
unique_counts = normal_df[final_features].nunique()
columns_with_only_one_value = list(unique_counts[unique_counts == 1].index)

# %%
for feature in set(final_features) - set(columns_with_only_one_value):
    print(normal_df.loc[indices][feature].value_counts(dropna=False))
    print("\n")

# %% [markdown]
# Here are the images that will go into the questionaire.

# %%
list(normal_df.loc[indices]['Filename'])

# %% [markdown]
# ### Next steps
#  1. The classes attributed to each image were attributed by a single human. We could iterate on that and have multiple people classify these images to avoid bias.
#  2. We could create multiple questionaires isolating different labels, to
#  better measure the effect of each label class on the human perception of the car.

# %%
