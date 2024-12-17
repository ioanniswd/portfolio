# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: myenv
#     language: python
#     name: myenv
# ---

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools

# %%
df = pd.read_excel('data.xlsx')

# %%
df.head(3)

# %%
df.info()

# %%
df['Holiday'].value_counts(dropna=False)

# %%
df['Seasons'].value_counts(dropna=False)

# %%
df.describe()

# %%
df[(df['Hour'] >= 8) & (df['Hour'] <= 17)]

# %%
df[(df['Hour'] >= 8) & (df['Hour'] <= 17)]['Rented Bike Count'].describe()

# %%
df.isna().sum()

# %%
# Keep only working hours and functioning days
# If maintenance is possible in non-functioning days, it can be performed at
# will, no need to check for the least busy hour
__df = df.copy()[(df['Hour'] >= 8) & (df['Hour'] <= 17) & (df['Functioning Day'] == 'Yes')]

# %%
__df.groupby('Hour')['Rented Bike Count'].mean()

# %%
__df.groupby('Hour')['Rented Bike Count'].std()

# %%
__df.groupby(['Seasons', 'Hour'])['Rented Bike Count'].mean()

# %%
import seaborn as sns
sns.set_theme(style="darkgrid")
colorblind_palette = sns.color_palette("colorblind")

# %%
# show current sns colors
sns.color_palette()

# %%
sns.color_palette().as_hex()[0:4]

# %%
values = sns.color_palette().as_hex()[0:4]
keys = ['Winter', 'Autumn', 'Spring', 'Summer']

season_colors = dict(zip(keys, values))

# %%
sns.color_palette("colorblind")

# %%
values = sns.color_palette('colorblind').as_hex()[0:4]
keys = ['Winter', 'Autumn', 'Spring', 'Summer']

colorblind_colors = dict(zip(keys, values))

# %%
ordered_seasons = ['Autumn', 'Winter', 'Spring', 'Summer']

# %%
viz_df = __df.groupby(['Seasons', 'Hour'])['Rented Bike Count'].mean().reset_index()
viz_df['Seasons'] = pd.Categorical(viz_df['Seasons'], categories=ordered_seasons, ordered=True)

fig, ax = plt.subplots(figsize=(12, 6))

sns.barplot(x='Hour',
            y='Rented Bike Count',
            data=viz_df,
            hue='Seasons',
            color='Seasons',
            palette=season_colors,
            ax=ax
            )

ax.set_title('Average Rented Bike Count by Hour and Season')
ax.set_ylabel('Average Rented Bike Count')
plt.show()

# %%
viz_df = __df.groupby(['Seasons', 'Hour'])['Rented Bike Count'].mean().reset_index()
viz_df['Seasons'] = pd.Categorical(viz_df['Seasons'], categories=ordered_seasons, ordered=True)

fig, ax = plt.subplots(figsize=(12, 6))

sns.barplot(x='Hour',
            y='Rented Bike Count',
            data=viz_df,
            hue='Seasons',
            color='Seasons',
            palette=colorblind_colors,
            ax=ax
            )

ax.set_title('Average Rented Bike Count by Hour and Season')
ax.set_ylabel('Average Rented Bike Count')
plt.show()

# %% [markdown]
# Some questions to answer:
#  1. How long does maintenance take? Should I find a window of more than 3 hours?
#  2. Does the above change if we exclude holidays?

# %%
df['Holiday'].value_counts(dropna=False)

# %%
__df = df.copy()[(df['Hour'] >= 8) & (df['Hour'] <= 17) & (df['Functioning Day'] == 'Yes') & (df['Holiday'] == 'No Holiday')]

# %%
viz_df = __df.groupby(['Seasons', 'Hour'])['Rented Bike Count'].mean().reset_index()
viz_df['Seasons'] = pd.Categorical(viz_df['Seasons'], categories=ordered_seasons, ordered=True)

fig, ax = plt.subplots(figsize=(12, 6))

sns.barplot(x='Hour',
            y='Rented Bike Count',
            data=viz_df,
            hue='Seasons',
            color='Seasons',
            palette=season_colors,
            ax=ax
            )

ax.set_title('Average Rented Bike Count by Hour and Season')
ax.set_ylabel('Average Rented Bike Count')
plt.show()

# %%
viz_df = __df.groupby(['Seasons', 'Hour'])['Rented Bike Count'].mean().reset_index()
viz_df['Seasons'] = pd.Categorical(viz_df['Seasons'], categories=ordered_seasons, ordered=True)

fig, ax = plt.subplots(figsize=(12, 6))

sns.barplot(x='Hour',
            y='Rented Bike Count',
            data=viz_df,
            hue='Seasons',
            color='Seasons',
            palette=colorblind_colors,
            ax=ax
            )

ax.set_title('Average Rented Bike Count by Hour and Season')
ax.set_ylabel('Average Rented Bike Count')
plt.show()

# %%
cmap = plt.cm.Oranges

# %%
# a heatmap of the average rented bike count by hour and season
viz_df = __df.groupby(['Seasons', 'Hour'])['Rented Bike Count'].mean().reset_index()
viz_df['Seasons'] = pd.Categorical(viz_df['Seasons'], categories=ordered_seasons, ordered=True)

# two axes
fig, ax = plt.subplots(figsize=(12, 6))

# a heatmap of the average rented bike count, with x = hour and y = season
sns.heatmap(viz_df.pivot(index='Seasons', columns='Hour', values='Rented Bike Count'),
            annot=True,
            fmt='.0f',
            cmap=cmap,
            ax=ax
            )

ax.yaxis.set_tick_params(rotation=0)


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


# %%
# a heatmap of the average rented bike count by hour and season
viz_df = __df.groupby(['Seasons', 'Hour'])['Rented Bike Count'].mean().reset_index()
viz_df['Seasons'] = pd.Categorical(viz_df['Seasons'], categories=ordered_seasons, ordered=True)
viz_df['Rented Bike Count'] = viz_df['Rented Bike Count'].round(0).astype(int)

fig, ax = plt.subplots(figsize=(12, 6), dpi=300)


vmin=viz_df['Rented Bike Count'].max() * -1
# Using a larger maximum to avoid having annotations with black color, which we cannot so easily fade out of the viewers attention
# vmax=viz_df['Rented Bike Count'].max() * 1.40
vmax=viz_df['Rented Bike Count'].max()

# a heatmap of the average rented bike count, with x = hour and y = season
sns.heatmap(viz_df.pivot(index='Seasons', columns='Hour', values='Rented Bike Count'),
            annot=True,
            fmt='.0f',
            cmap=cmap,
            ax=ax,
            vmin=vmin,
            vmax=vmax
            )

ax.yaxis.set_tick_params(rotation=0)
ax.set_xlabel('Hour of Day')

rectangles_color = '#0C8040'

rect_options = {
  'width': 3,
  'height': 1,
  'fill': False,
  'edgecolor': rectangles_color,
  'lw': 3,
  'clip_on': False
}

# add a rectangle
rect1 = plt.Rectangle(xy=(1, 0), **rect_options)
ax.add_patch(rect1)

rect2 = plt.Rectangle(xy=(1, 2), **rect_options)
ax.add_patch(rect2)

rect3 = plt.Rectangle(xy=(2, 3), **rect_options)
ax.add_patch(rect3)

# Side texts
text_options = {
  'x': 0,
  'y': 0,
  'ha': 'left',
  'va': 'bottom',
  'fontsize': 14,
  'linespacing': 1.5
}

side_texts_height = 0

t1 = ax.text(
  s='$\\bf{Start\\ and\\ End\\ of\\ day}$\nSeem to be the busiest times for bike rental.\nBest to avoid maintenance before 9am and after 4pm.\nThis seems to be consistent for all seasons.',
  # color='#F79747',
  color='#6C0E23',
  **text_options
  )

side_texts_height += get_text_coordinates(t1, ax=ax, fig=fig)['y1']

t2 = ax.text(
  s='$\\bf{Winter}$\nMaintenance window becomes obsolete as much\nfewer bikes are rented. Could avoid 8am.This season is\nideal for general overhauls lasting longer than simple\nmaintenance.',
  color=cmap(0.5),
  **text_options
  )

side_texts_height += get_text_coordinates(t2, ax=ax, fig=fig)['y1']

t3 = ax.text(
  s='$\\bf{Rest\\ of\\ the\\ Year}$\nThe timeframe between 9am and 11am is ideal for\nmaintenance during Autumn and Spring.\nDuring the Summer months, the ideal window is\nshifted by one hour, to between 10am and 12am.',
  color=rectangles_color,
  **text_options
  )

side_texts_height += get_text_coordinates(t3, ax=ax, fig=fig)['y1']

# side_texts_start_y0 = 0.7
side_texts_start_y0 = 1
# side_texts_start_y0 = 1.7
# side_texts_start_y0 = 1.2
side_texts_end_y1 = 3

spacing = (side_texts_end_y1 - side_texts_start_y0 - side_texts_height) / (3 - 1) # n_elements - 1

side_box_x = 12.5

# t1.set_position((side_box_x, side_texts_start_y0 + spacing))
t1.set_position((side_box_x, side_texts_start_y0))

y1 = get_text_coordinates(t1, ax=ax, fig=fig)['y1'] + spacing

t2.set_position((side_box_x, y1))

y2 = get_text_coordinates(t2, ax=ax, fig=fig)['y1'] + spacing

t3.set_position((side_box_x, y2))

title = ax.text(
  x=-1,
  y=-0.35,
  s='Average Rented Bike Count by Hour and Season',
  fontsize=18,
  ha='left',
  va='top',
  weight='bold'
)

plt.show()

# %% [markdown]
# The above can be included in the written report.
# <br />
# The following will be included in the presentation.

# %%
# a heatmap of the average rented bike count by hour and season
viz_df = __df.groupby(['Seasons', 'Hour'])['Rented Bike Count'].mean().reset_index()
viz_df['Seasons'] = pd.Categorical(viz_df['Seasons'], categories=ordered_seasons, ordered=True)
viz_df['Rented Bike Count'] = viz_df['Rented Bike Count'].round(0).astype(int)

fig, ax = plt.subplots(figsize=(12, 6))

data = viz_df.pivot(index='Seasons', columns='Hour', values='Rented Bike Count')

# vmin=viz_df['Rented Bike Count'].min()
vmin=viz_df['Rented Bike Count'].max() * -1
# Using a larger maximum to avoid having annotations with black color, which we cannot so easily fade out of the viewers attention
# vmax=viz_df['Rented Bike Count'].max() * 1.40
vmax=viz_df['Rented Bike Count'].max()

# a heatmap of the average rented bike count, with x = hour and y = season
sns.heatmap(data,
            annot=True,
            fmt='.0f',
            cmap=cmap,
            ax=ax,
            cbar=False,
            vmin=vmin,
            vmax=vmax
            )

title = ax.text(
  x=-1,
  y=-0.35,
  s='Average Rented Bike Count by Hour and Season (Overview)',
  fontsize=18,
  ha='left',
  va='top',
  weight='bold'
)

ax.yaxis.set_tick_params(rotation=0)
ax.set_xlabel('Hour of Day')

# remove gray grid
ax.grid(False)

# %% [markdown]
# ### Peak times

# %%
# a heatmap of the average rented bike count by hour and season
viz_df = __df.groupby(['Seasons', 'Hour'])['Rented Bike Count'].mean().reset_index()
viz_df['Seasons'] = pd.Categorical(viz_df['Seasons'], categories=ordered_seasons, ordered=True)
viz_df['Rented Bike Count'] = viz_df['Rented Bike Count'].round(0).astype(int)

fig, ax = plt.subplots(figsize=(12, 6))

data = viz_df.pivot(index='Seasons', columns='Hour', values='Rented Bike Count')

mask = np.zeros_like(data, dtype=bool)
alpha = 0.4
vmin=viz_df['Rented Bike Count'].min()
vmin=viz_df['Rented Bike Count'].max() * -1
# Using a larger maximum to avoid having annotations with black color, which we cannot so easily fade out of the viewers attention
# vmax=viz_df['Rented Bike Count'].max() * 1.40
vmax=viz_df['Rented Bike Count'].max()


highlight_coords = list(itertools.product(
            list(range(0,4)),
            [0] + list(range(len(viz_df['Hour'].unique()) - 2, len(viz_df['Hour'].unique())))
            ))

for coord in highlight_coords:
    mask[coord] = True

# a heatmap of the average rented bike count, with x = hour and y = season
sns.heatmap(data,
            mask=mask,
            annot=True,
            fmt='.0f',
            cmap=cmap,
            ax=ax,
            alpha=alpha,
            cbar=False,
            vmin=vmin,
            vmax=vmax
            )

title = ax.text(
  x=-1,
  y=-0.35,
  s='Average Rented Bike Count by Hour and Season (Peak times)',
  fontsize=18,
  ha='left',
  va='top',
  weight='bold'
)

# decrease opacity of the x-axis labels
for label in ax.get_xticklabels():
    if label.get_text() in ['8', '16', '17']:
      # make bold
      label.set_weight('bold')
      continue

    label.set_alpha(alpha)

mask = np.zeros_like(data, dtype=bool)

for coord in highlight_coords:
    mask[coord] = True

sns.heatmap(data,
            mask=~mask,
            annot=True,
            fmt='.0f',
            cmap=cmap,
            ax=ax,
            cbar=False,
            # alpha=1,
            vmin=vmin,
            vmax=vmax
            )

ax.yaxis.set_tick_params(rotation=0)
ax.set_xlabel('Hour of Day')

# remove gray grid
ax.grid(False)

# %% [markdown]
# ### Winter

# %%
# a heatmap of the average rented bike count by hour and season
viz_df = __df.groupby(['Seasons', 'Hour'])['Rented Bike Count'].mean().reset_index()
viz_df['Seasons'] = pd.Categorical(viz_df['Seasons'], categories=ordered_seasons, ordered=True)
viz_df['Rented Bike Count'] = viz_df['Rented Bike Count'].round(0).astype(int)

fig, ax = plt.subplots(figsize=(12, 6))

data = viz_df.pivot(index='Seasons', columns='Hour', values='Rented Bike Count')

mask = np.zeros_like(data, dtype=bool)
alpha = 0.4

highlight_coords = list(itertools.product(
            [1],
            range(0, len(viz_df['Hour'].unique())
            )))

for coord in highlight_coords:
    mask[coord] = True

# a heatmap of the average rented bike count, with x = hour and y = season
sns.heatmap(data,
            mask=mask,
            annot=True,
            fmt='.0f',
            cmap=cmap,
            ax=ax,
            alpha=alpha,
            cbar=False,
            vmin=vmin,
            vmax=vmax
            )

title = ax.text(
  x=-1,
  y=-0.35,
  s='Average Rented Bike Count by Hour and Season (Winter)',
  fontsize=18,
  ha='left',
  va='top',
  weight='bold'
)

mask = np.zeros_like(data, dtype=bool)

for coord in highlight_coords:
    mask[coord] = True

sns.heatmap(data,
            mask=~mask,
            annot=True,
            fmt='.0f',
            cmap=cmap,
            ax=ax,
            cbar=False,
            # alpha=1,
            vmin=vmin,
            vmax=vmax
            )

ax.yaxis.set_tick_params(rotation=0)
ax.set_xlabel('Hour of Day')

for label in ax.get_yticklabels():
    if label.get_text() in ['Winter']:
      # make bold
      label.set_weight('bold')
      continue

    # Need it to be a little more transparent
    label.set_alpha(alpha)

# remove gray grid
ax.grid(False)

# %% [markdown]
# ### Autumn and Spring

# %%
# a heatmap of the average rented bike count by hour and season
viz_df = __df.groupby(['Seasons', 'Hour'])['Rented Bike Count'].mean().reset_index()
viz_df['Seasons'] = pd.Categorical(viz_df['Seasons'], categories=ordered_seasons, ordered=True)
viz_df['Rented Bike Count'] = viz_df['Rented Bike Count'].round(0).astype(int)

fig, ax = plt.subplots(figsize=(12, 6))

data = viz_df.pivot(index='Seasons', columns='Hour', values='Rented Bike Count')

mask = np.zeros_like(data, dtype=bool)
alpha = 0.4

# highlight_coords = [(1, 0), (1, 2), (2, 3)]
highlight_coords = list(itertools.product(
            [0, 2],
            [1, 2, 3]
            ))

for coord in highlight_coords:
    mask[coord] = True

# a heatmap of the average rented bike count, with x = hour and y = season
sns.heatmap(data,
            mask=mask,
            annot=True,
            fmt='.0f',
            cmap=cmap,
            ax=ax,
            alpha=alpha,
            cbar=False,
            vmin=vmin,
            vmax=vmax
            )

title = ax.text(
  x=-1,
  y=-0.35,
  s='Average Rented Bike Count by Hour and Season (Autumn/Spring maintenance)',
  fontsize=18,
  ha='left',
  va='top',
  weight='bold'
)

# decrease opacity of the x-axis labels
for label in ax.get_xticklabels():
    if label.get_text() in ['9', '10', '11']:
      # make bold
      label.set_weight('bold')
      continue

    label.set_alpha(alpha)

mask = np.zeros_like(data, dtype=bool)

for coord in highlight_coords:
    mask[coord] = True

sns.heatmap(data,
            mask=~mask,
            annot=True,
            fmt='.0f',
            cmap=cmap,
            ax=ax,
            cbar=False,
            # alpha=1,
            vmin=vmin,
            vmax=vmax
            )

ax.yaxis.set_tick_params(rotation=0)
ax.set_xlabel('Hour of Day')

for label in ax.get_yticklabels():
    if label.get_text() in ['Autumn', 'Spring']:
      # make bold
      label.set_weight('bold')
      continue

    label.set_alpha(alpha)

# remove gray grid
ax.grid(False)

# %% [markdown]
# ### Summer

# %%
# a heatmap of the average rented bike count by hour and season
viz_df = __df.groupby(['Seasons', 'Hour'])['Rented Bike Count'].mean().reset_index()
viz_df['Seasons'] = pd.Categorical(viz_df['Seasons'], categories=ordered_seasons, ordered=True)
viz_df['Rented Bike Count'] = viz_df['Rented Bike Count'].round(0).astype(int)

fig, ax = plt.subplots(figsize=(12, 6))

data = viz_df.pivot(index='Seasons', columns='Hour', values='Rented Bike Count')

mask = np.zeros_like(data, dtype=bool)

# highlight_coords = [(1, 0), (1, 2), (2, 3)]
highlight_coords = list(itertools.product(
            [3],
            [2, 3, 4]
            ))

for coord in highlight_coords:
    mask[coord] = True

# a heatmap of the average rented bike count, with x = hour and y = season
sns.heatmap(data,
            mask=mask,
            annot=True,
            fmt='.0f',
            cmap=cmap,
            ax=ax,
            alpha=alpha,
            cbar=False,
            vmin=vmin,
            vmax=vmax
            )

title = ax.text(
  x=-1,
  y=-0.35,
  s='Average Rented Bike Count by Hour and Season (Summer maintenance)',
  fontsize=18,
  ha='left',
  va='top',
  weight='bold'
)

# decrease opacity of the x-axis labels
for label in ax.get_xticklabels():
    if label.get_text() in ['10', '11', '12']:
      # make bold
      label.set_weight('bold')
      continue

    label.set_alpha(alpha)

mask = np.zeros_like(data, dtype=bool)

for coord in highlight_coords:
    mask[coord] = True

sns.heatmap(data,
            mask=~mask,
            annot=True,
            fmt='.0f',
            cmap=cmap,
            ax=ax,
            cbar=False,
            # alpha=1,
            vmin=vmin,
            vmax=vmax
            )

ax.yaxis.set_tick_params(rotation=0)
ax.set_xlabel('Hour of Day')

for label in ax.get_yticklabels():
    if label.get_text() in ['Summer']:
      # make bold
      label.set_weight('bold')
      continue

    label.set_alpha(alpha)

# remove gray grid
ax.grid(False)

# %%
