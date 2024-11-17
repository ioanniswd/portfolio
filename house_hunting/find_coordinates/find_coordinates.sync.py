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
import requests
import seaborn as sns
import plotly.express as px
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import time

# %%
file_prefix = '../data/schools_attica'

# %%
df = pd.read_csv(f'{file_prefix}.csv')
df

# %%
# remove the ΤΚ from the postal code
df['Διεύθυνση'] = df['Διεύθυνση'].str.replace('ΤΚ', '')

# %%
# replace multiple spaces with a single space
df['Διεύθυνση'] = df['Διεύθυνση'].str.replace(' +', ' ')

# %%
df

# %%
overpass_url = "http://overpass-api.de/api/interpreter"


# %%
def get_coordinates(address):
    overpass_query = f"""
    [out:json];
    node["addr:street"="{address}"];
    out center;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()

    if len(data['elements']) == 0:
        return None, None

    return data['elements'][0]['lat'], data['elements'][0]['lon']


# %%
for (idx, address) in df['Διεύθυνση'].tail(20).items():
    lat, lon = get_coordinates(address)
    df.loc[idx, 'lat'] = lat
    df.loc[idx, 'lon'] = lon

# %%
with open('../google_maps_api_key') as f:
    api_key = f.read()

# %%
url = f'https://maps.googleapis.com/maps/api/geocode/json'

# %%
# get the coordinates using google maps api
for (idx, address) in df['Διεύθυνση'].head(3).items():
# for (idx, address) in df['Διεύθυνση'].items():
    if pd.notnull(df.loc[idx, 'lat']) or pd.notnull(df.loc[idx, 'lon']):
        continue

    search_params = {
        'address': address,
        'key': api_key
    }

    response = requests.get(url, params=search_params)
    data = response.json()
#     print(data)
    try:
        lat = data['results'][0]['geometry']['location']['lat']
        lon = data['results'][0]['geometry']['location']['lng']
        df.loc[idx, 'lat'] = lat
        df.loc[idx, 'lon'] = lon
    except:
        print('Error in response')
        print('Response: ', data)
        print('Address: ', address)
        print('Moving on...')
        print("\n")

# %%
df

# %%
df.isna().sum()

# %%
len(df)

# %%
pd.set_option('display.max_rows', 100)

# %%
df[(df['lat'].isna()) & (~df['Διεύθυνση'].isna())]

# %%
# address = 'ΠΑΠΑΦΛΕΣΣΑ 6, 13461'
address = 'Papaflessa 6, 13461'

# %%
search_params = {
    'address': address,
    'key': api_key
}

response = requests.get(url, params=search_params)
data = response.json()

# %%
data

# %% [markdown]
# There is a different postal code for the above address. I can only assume there is some error in the inputed data. Any address not found will have to be fixed by hand.

# %%
df.tail()

# %% [markdown]
# Address `Μεγάλου Αλεξάνδρου & Κιθαιρώνος, 19012` was found. For now, we can keep only the schools whose coordinates we know.

# %%
df['Κατηγορία'].value_counts()

# %%
df['Type'] = np.where(df['Κατηγορία'] == 'Ιδιωτικά Σχολεία', 'Private', 'Public')

# %%
# df.to_csv(f'{file_prefix}_coordinates.csv', index=False)

# %%
df = pd.read_csv(f'{file_prefix}_coordinates.csv')

# %%
df.isna().sum()

# %%

# %%
# create a large plotly express map graph
fig = px.scatter_mapbox(
    df,
    lat='lat',
    lon='lon',
    hover_name='Σχολείο',
    # color='Κατηγορία'
    )

fig.update_layout(
        mapbox_style='open-street-map',
        margin=dict(l=20, r=20, t=20, b=20),
        height=1000,
        width=1400
        )

fig.show()

# %%
# create a large plotly express map graph
fig = px.scatter_mapbox(
    df,
    lat='lat',
    lon='lon',
    hover_name='Σχολείο',
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
