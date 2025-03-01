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
df = df[df['loan_status'].isin(['Default', 'Charged Off', 'Fully Paid'])]
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
columns_to_keep.add('disbursement_method')

# %%
columns_to_keep

# %%

# %%

# %%

# %%
columns_to_drop = set()

# %%
print(column_definition('emp_title'))

# %%
columns_to_drop.add('emp_title')

# %%
df['desc'].dropna().head()

# %% [markdown]
# Same with 'desc`.

# %%
columns_to_drop.add('desc')

# %%
df.groupby(['purpose', 'title']).size()

# %%
df['purpose'].value_counts()

# %%
df['title'].nunique()

# %% [markdown]
# Same for `title`

# %%
columns_to_drop.add('title')

# %% [markdown]
# Let's just use the state, using the first 3 digits of the zip code seems a bit overkill for this first iteration.

# %%
columns_to_drop.add('zip_code')

# %%
print(column_definition('dti'))

# %% [markdown]
# Seems useful, let's keep it.

# %%
print(column_definition('delinq_2yrs'))

# %% [markdown]
# Same

# %%
print(column_definition('earliest_cr_line'))

# %% [markdown]
# We could keep the month of year and the year as features. Will do later on.

# %%
print(column_definition('initial_list_status'))

# %%
print(column_definition('out_prncp'))

# %%
print(column_definition('out_prncp_inv'))

# %%
columns_to_drop.add('out_prncp_inv')

# %%
print(column_definition('total_pymnt_inv'))

# %%
columns_to_drop.add('total_pymnt_inv')

# %%
