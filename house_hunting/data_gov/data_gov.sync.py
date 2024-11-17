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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import requests
import plotly.express as px
import pandas as pd
import numpy as np

# %%
with open('../keys/data_gov_api_key', 'r') as f:
    api_key = f.read().strip()

# %%
# https://data.gov.gr/datasets/minedu_schools/
url = 'https://data.gov.gr/api/v1/query/minedu_schools'
headers = {'Authorization':f'Token {api_key}'}
response = requests.get(url, headers=headers)
response.json()[0:5]

# %%
len(response.json())

# %%
df = pd.DataFrame(response.json())
df.head()

# %%
df.info()

# %%
df.isna().sum().sort_values(ascending=False)

# %%
df['area'].value_counts()

# %%
df['municipality'].value_counts()

# %%
df['district'].value_counts()

# %%
df['regional_unit'].value_counts()

# %%
df['school_type'].value_counts()

# %%
# create a large plotly express map graph
viz_df = df.copy()[df['regional_unit'] == 'ΠΕΡΙΦΕΡΕΙΑΚΗ Δ/ΝΣΗ Π/ΘΜΙΑΣ ΚΑΙ Δ/ΘΜΙΑΣ ΕΚΠ/ΣΗΣ ΑΤΤΙΚΗΣ']
viz_df['Type'] = np.where(viz_df['school_type'] == 'Ιδιωτικά Σχολεία', 'Private', 'Public')

fig = px.scatter_mapbox(
    viz_df,
    lat='lat',
    lon='lng',
    hover_name='school_name',
    color='Type'
    )

fig.update_layout(
        mapbox_style='open-street-map',
        margin=dict(l=20, r=20, t=20, b=20),
        height=1000,
        width=1400
        )

fig.show()

# %%
