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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import yfinance as yf

# %%
sns.set_theme(style='darkgrid')
sns.set(font='Noto Sans Mono')

# %%
TRADING_DAYS_IN_A_YEAR = 252
np.random.seed(42)  # For reproducibility

# %%
# Get data for multiple stocks
tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
# tickers = [
#         '^GSPC',  # S&P 500
#         'XOM',  # Exxon Mobil
#         'NEE',  # NextEra Energy
#         'ENPH'  # Enphase Energy
#         ]

data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Close']

# Compute daily returns
returns = data.pct_change().dropna()

# Compute correlation matrix
correlation_matrix = returns.corr()
# print(correlation_matrix)

# %%
fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        ax=ax
        )

ax.set_title(
        'Stock Returns Correlation Matrix',
        fontsize=15,
        fontweight='bold'
        )

plt.show()

# %%
returns

# %%
mean_returns = returns.mean().values * TRADING_DAYS_IN_A_YEAR
std_devs = returns.std().values * np.sqrt(TRADING_DAYS_IN_A_YEAR)
cov_matrix = returns.cov().values * TRADING_DAYS_IN_A_YEAR

# %%
# Let's assume a portfolio with equal weights
portfolio_weights = np.array([0.25, 0.25, 0.25, 0.25])
assert np.isclose(np.sum(portfolio_weights), 1), "Portfolio weights do not sum to 1"

# %%
n_simulations = 10_000  # Number of Monte Carlo simulations
n_assets = len(tickers)

simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_simulations)

portfolio_returns = np.dot(simulated_returns, portfolio_weights)

mean_portfolio_return = np.mean(portfolio_returns)
value_at_risk_95 = np.percentile(portfolio_returns, 5)  # 5th percentile (95% VaR)
loss_prob = np.mean(portfolio_returns < 0)  # Probability of loss

# %%
print(f'Mean Portfolio Return: {mean_portfolio_return:.4f}')
print(f'95% VaR: {value_at_risk_95:.4f}')
print(f'Probability of Loss: {loss_prob:.4f}')

# %%
fig, ax = plt.subplots(figsize=(10, 6))

sns.histplot(
        portfolio_returns,
        bins=50,
        kde=True,
        color='#4A81BF',
        ax=ax
        )

var_line = ax.axvline(
        value_at_risk_95,
        ymin=0,
        ymax=0.90,
        color='red',
        linestyle='dashed',
        label=f'95% VaR: {value_at_risk_95:.2%}'
        )

ax.text(
        x=value_at_risk_95,
        y=0.992 * ax.get_ylim()[1],
        s=f'95% VaR:\n{value_at_risk_95:.2%}',
        color='red',
        ha='center',
        va='bottom'
        )

ax.axvline(
        mean_portfolio_return,
        ymin=0,
        ymax=0.90,
        color='green',
        linestyle='dashed',
        label=f'Mean Return: {mean_portfolio_return:.2%}'
        )

ax.text(
        x=mean_portfolio_return,
        y=0.992 * ax.get_ylim()[1],
        s=f'Mean Return:\n{mean_portfolio_return:.2%}',
        color='green',
        ha='center',
        va='bottom'
        )

ax.set_xlabel("Simulated Portfolio Return")
ax.set_ylabel("Frequency")

# hide legend
ax.legend().set_visible(False)

ax.set_title(
        "Monte Carlo Simulation of Portfolio Returns",
        fontsize=15,
        fontweight='bold'
        )

# raise ylim
ax.set_ylim([0, 1.1 * ax.get_ylim()[1]])


plt.show()

# %%
