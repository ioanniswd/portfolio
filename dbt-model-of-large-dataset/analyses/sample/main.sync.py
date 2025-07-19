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
import duckdb

# %%
duckdb.sql('SET enable_progress_bar = true')

# %%
# Load the data
duckdb.sql('''
           select
             *
           from '../../data/taxi_trips.csv'
           limit 10
           '''
).df().T


# %%
def qw(query):
    """
    Helper function to execute a query and return the result as a DataFrame.
    """
    with duckdb.connect('../../taxi_data.duckdb') as con:
        return con.execute(query).fetchdf()
    with duckdb.connect('../../taxi_data.duckdb') as con:
        df = con.execute(query).fetchdf()

    return df


# %%
qw('''
SELECT * FROM dbt_models.mart_trip_summary
''')

# %%
qw('''
SELECT * FROM dbt_models.payment_types
''')

# %%
qw('''
SELECT * FROM dbt_models.payment_types
''')

# %%
qw('''
SELECT *
FROM dbt_models.ratecodes
''')

# %%
qw('SELECT * FROM information_schema.tables')

# %%
qw('''
   SELECT
     taxi_trips.pickup_datetime,
     taxi_trips.trip_distance,
     ratecodes.name AS rate_code_name,
     payment_types.name AS payment_type_name
   FROM dbt_models.stg_taxi_trips AS taxi_trips
   LEFT JOIN dbt_models.ratecodes AS ratecodes
          ON taxi_trips.rate_code_id = ratecodes.id
   LEFT JOIN dbt_models.payment_types AS payment_types
          ON taxi_trips.payment_type = payment_types.id
   LIMIT 10
   ''')

# %%
qw('''
   SELECT
     taxi_trips.rate_code_id,
     ratecodes.name AS rate_code_name,
     count(*)
   FROM dbt_models.stg_taxi_trips AS taxi_trips
   LEFT JOIN dbt_models.ratecodes AS ratecodes
          ON taxi_trips.rate_code_id = ratecodes.id
   GROUP BY 1,2
   ORDER BY 1,2
   ''')

# %%
qw('''
   select *
   from dbt_models.stg_taxi_trips as taxi_trips
   where pickup_datetime > dropoff_datetime
   limit 100
   ''').T

# %%
qw('describe dbt_models.stg_taxi_trips')

# %%
qw('''
SELECT * FROM dbt_models.mart_distance_distribution
   ''')

# %%
qw('''
SELECT * FROM dbt_models.mart_distance_distribution_km
   ''')

# %%
