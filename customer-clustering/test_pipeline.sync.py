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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import joblib
from pathlib import Path
import pandas as pd
from preprocessing.preprocessor import Preprocessor

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
model_dir = Path('models')
model_filename = model_dir / 'customer_clustering_pipeline.joblib'

# %%
pipeline = joblib.load(model_filename)

# %%
original_df = pd.read_csv('data/marketing_campaign.csv', sep='\t')
new_data = original_df[-10:].copy()

# %%
new_data['Cluster'] = pipeline.predict(new_data)
new_data['Cluster'].value_counts()

# %%
