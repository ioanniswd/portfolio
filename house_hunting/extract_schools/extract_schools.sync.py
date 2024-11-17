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
import os
import pandas as pd
import numpy as np
import webbrowser
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from time import sleep

# %%
home_folder = os.path.expanduser("~")

# %%
options = Options()

# options.headless = True  # Run in headless mode
options.headless = False

options.set_preference("general.useragent.override", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0")

# Setup the Firefox WebDriver
service = Service(f'{home_folder}/geckodriver')  # Update the path to your geckodriver
browser = webdriver.Firefox(service=service, options=options)

# Remove the WebDriver property using JavaScript
browser.execute_script("""
    Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined
    });
""")

# %%
browser.get('https://keaprogram.gr/pubnr/MySchool')

# %%
table = browser.find_element(By.XPATH, "//table")

# %%
# find a tag elements under td
table_data = table.find_elements(By.XPATH, "//td/a")

# %%
table_data

# %%
table_data[0]

# %%
table_data[0].text

# %%
table_data[1].text

# %%
for data in table_data:
    print(data.text)

# %%
attica_administrations = [data for data in table_data if 'ΑΤΤΙΚΗ' in data.text or 'ΑΘΗΝΑ' in data.text]
attica_administrations

# %%
attica_administrations[0]

# %%
# print tag name for the first element
print(attica_administrations[0].tag_name)

# %%
# find url for the first element
attica_administrations[0].get_attribute('href')

# %%
districts = []

for administration in attica_administrations:
    districts.append({
        'name': administration.text,
        'url': administration.get_attribute('href')
        })

# %%
# Open the first administration
browser.get(attica_administrations[0].get_attribute('href'))

# %%
# Find the table with schools
schools_table = browser.find_element(By.XPATH, "//table")

# %%
num_of_columns = len(schools_table.find_elements(By.XPATH, "//thead/tr/td"))
num_of_columns

# %%
# get all the data from the table
data = [x.text for x in schools_table.find_elements(By.XPATH, "//td")]

# %%
# reshape
data = np.array(data).reshape(-1, num_of_columns)

# %%
data

# %%
df = pd.DataFrame()

# %%
columns = data[0]
data = data[1:]
df = pd.concat([df, pd.DataFrame(data, columns=columns)])

# %%
df

# %%
df = pd.DataFrame()

for district in districts:
    browser.get(district['url'])
    sleep(3)

    schools_table = browser.find_element(By.XPATH, "//table")
    num_of_columns = len(schools_table.find_elements(By.XPATH, "//thead/tr/td"))
    data = [x.text for x in schools_table.find_elements(By.XPATH, "//td")]
    data = np.array(data).reshape(-1, num_of_columns)
    columns = data[0]
    data = data[1:]
    __df = pd.DataFrame(data, columns=columns)
    __df['Περιφερειακή Διεύθυνση'] = district['name']
    df = pd.concat([df, __df])

# %%
df

# %%
df.drop('#', axis=1, inplace=True)

# %%
df['Κατηγορία'].value_counts()

# %%
df[df['Κατηγορία'] == 'Ιδιωτικά Σχολεία']

# %%
df[(df['Κατηγορία'] == 'Ιδιωτικά Σχολεία') & (df['Κατηγορία'].str.upper().str.contains('ΦΡΟΝΤΙΣΤ'))]

# %%
pd.set_option('display.max_rows', None)

# %%
df[df['Κατηγορία'] == 'Ιδιωτικά Σχολεία']

# %% [markdown]
# A bare minimum for an address would include at least `Οδός Αριθμός, ΤΚ`.
# `ΤΚ00000` -> 7 characters, at least 1 digit for street number and at least 3
# digits for street name.
# Any address with less than 7 characters is not a valid address.

# %%
df[df['Διεύθυνση'].str.len() < 7]

# %%
df['Διεύθυνση'] = np.where(df['Διεύθυνση'].str.len() < 7, np.nan, df['Διεύθυνση'])

# %%
df.isna().sum()

# %%
len(df)

# %%
df[df['Διεύθυνση'].isna()]['Κατηγορία'].value_counts()

# %%
df[(df['Διεύθυνση'].isna()) & (df['Κατηγορία'] == 'Γυμνάσια')]

# %%
df[df['Διεύθυνση'].isna()]['Περιφερειακή Διεύθυνση'].value_counts()

# %%
df['Κατηγορία'].value_counts()

# %%
df['Περιφερειακή Διεύθυνση'].value_counts()

# %%
df.to_csv('../data/schools_attica.csv', index=False)

# %%
