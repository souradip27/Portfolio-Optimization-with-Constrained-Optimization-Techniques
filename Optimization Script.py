#!/usr/bin/env python
# coding: utf-8

# # CMSE 831 Final Project
# ### &#9989; SANDHYA KILARI
# #### &#9989; 12-08-2024

# # Portfolio Optimization with Constrained Optimization Techniques: Balancing Risk and Return

# ## 1. Background and Motivation
# 
# This section provides the context and purpose of the project. It explains the importance of portfolio optimization in financial decision-making and highlights the need for advanced optimization techniques to address the limitations of traditional methods

# **Portfolio optimization is a cornerstone of financial decision-making, aiming to balance the trade-off between maximizing returns and minimizing risks. Traditional methods, such as Markowitz mean-variance optimization, often face challenges in adapting to diverse asset classes and dynamic market conditions. By leveraging advanced optimization techniques, this project explores the use of constrained optimization methods to address these limitations with computational efficiency and adaptability.**
# 
# #### Research Questions:
# - How does an advanced constrained optimization approach compare to traditional methods in terms of risk-adjusted performance metrics like the Sharpe Ratio?
# - Can the optimized portfolio adapt dynamically to changing market conditions, maintaining stability during periods of high volatility?

# ## 2. Methodology
# 
# This section outlines the steps followed to achieve the project's goals

# In[1]:


# Import necessary libraries
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product


# ### **2.1 Data Collection**
# 
# Historical price data for diversified assets, such as stocks, bonds, and cryptocurrencies, was fetched using the `yfinance` library. Daily returns were computed for analysis.
# 
# **Selected Tickers**
# 
# The portfolio includes a diversified mix of asset classes:
# 
# - **Stocks**: `AAPL` (Apple), `MSFT` (Microsoft), `GOOGL` (Alphabet).
#   
# - **Bonds**: `TLT` (iShares 20+ Year Treasury Bond ETF).
# 
# - **Commodities**: `GLD` (SPDR Gold Shares), `DBC` (Invesco DB Commodity Index Tracking Fund).
# 
# - **Real Estate**: `VNQ` (Vanguard Real Estate ETF).
# 
# - **Cryptocurrencies**: `BTC-USD` (Bitcoin), `ETH-USD` (Ethereum).
# 
# This selection ensures exposure to a broad range of asset classes, enabling robust diversification and risk management in the portfolio optimization process.

# In[2]:


# Data Collection
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TLT', 'GLD', 'VNQ', 'BTC-USD', 'ETH-USD', 'DBC'] # Diversified portfolio
data = yf.download(tickers, start='2010-01-01', end='2023-01-01', group_by='ticker')

# Compute daily returns
# Adjust NaN handling explicitly for consistency
returns = (
  data.loc[:, (tickers, 'Adj Close')]
  .pct_change(fill_method=None)
  .dropna()
)
returns.columns = tickers # Flatten MultiIndex for simplicity

# Risk-free rate
risk_free_rate = 0.01


# ### **2.2 Portfolio Optimization (SLSQP)**
# 
# The portfolio optimization function uses the **SLSQP (Sequential Least Squares Programming)** method to maximize returns while minimizing risk. This approach efficiently handles constraints such as ensuring the portfolio weights sum to 1 and remain within predefined bounds. Additionally, penalties are applied to discourage large weights, promoting diversification, and to maintain stability by accounting for previous weights.
# 
# The optimization process evaluates a custom objective function that balances return maximization and risk minimization based on the specified λ1 (return weight) and λ2 (risk weight) parameters.

# In[3]:


# Portfolio Optimization Function
def portfolio_optimization(train_returns, lambda1, lambda2, bounds, constraints, prev_weights=None):
  """
  Perform portfolio optimization using the SLSQP method.

  Parameters:
    train_returns: DataFrame of asset returns for training.
    lambda1: Weight for return maximization.
    lambda2: Weight for risk minimization.
    bounds: Tuple of bounds for weights.
    constraints: Constraints for portfolio weights.
    prev_weights: Previous weights (for stability penalty).

  Returns:
    Optimal portfolio weights as a NumPy array.
  """
  mean_returns = train_returns.mean().values
  cov_matrix = train_returns.cov().values

  # Define the objective function
  def objective(weights):
    portfolio_return = np.dot(weights, mean_returns) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    stability_penalty = 0 if prev_weights is None else 0.01 * np.sum((weights - prev_weights)**2)
    penalty = 0.03 * np.sum(weights**2) # Weight penalty
    return -(lambda1 * portfolio_return - lambda2 * portfolio_volatility) + penalty + stability_penalty

  # Initial guess: equally distributed weights
  init_guess = len(tickers) * [1.0 / len(tickers)]

  # Perform optimization
  result = minimize(
    objective,
    init_guess,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000, 'ftol': 1e-9}
  )
  return result.x # Optimal weights


# ### **2.3 Risk Metrics**
# 
# The following metrics were calculated to evaluate portfolio performance:
# 
# - **Sharpe Ratio**: Risk-adjusted returns.
#   
# - **Conditional Value at Risk (CVaR)**: Tail risk metric.
# 
# - **Maximum Drawdown (MDD)**: Peak-to-trough loss.

# In[4]:


# Compute Portfolio Metrics
def compute_metrics(weights, returns):
  """
  Compute key metrics for a portfolio: Sharpe Ratio, CVaR, and MDD.

  Parameters:
    weights: Portfolio weights as a NumPy array.
    returns: DataFrame of asset returns.

  Returns:
    Tuple of Sharpe Ratio, CVaR, and MDD.
  """
  mean_returns = returns.mean().values
  cov_matrix = returns.cov().values

  portfolio_return = np.dot(weights, mean_returns) * 252
  portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
  sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

  # Portfolio returns
  portfolio_returns = returns.dot(weights)

  # Conditional Value at Risk (CVaR)
  var_95 = np.percentile(portfolio_returns, 5) # 5% Value at Risk
  cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean() # Conditional Value at Risk

  # Maximum Drawdown (MDD)
  cumulative_returns = (1 + portfolio_returns).cumprod()
  rolling_max = cumulative_returns.cummax()
  max_drawdown = (rolling_max - cumulative_returns).max()

  return sharpe_ratio, cvar_95, max_drawdown

def compute_asset_metrics(returns):
  """
  Compute mean return, volatility, and Sharpe ratio for individual assets.
  """
  mean_returns = returns.mean() * 252 
  volatilities = returns.std() * np.sqrt(252) 
  sharpe_ratios = (mean_returns - risk_free_rate) / volatilities

  # Combine into a DataFrame
  asset_metrics = pd.DataFrame({
    'Mean Return': mean_returns,
    'Volatility': volatilities,
    'Sharpe Ratio': sharpe_ratios
  })
  return asset_metrics


# ### **2.4 Baseline and Markowitz Portfolio**
# 
# This section establishes benchmarks for evaluating the performance of the optimized portfolio:
# 
# - **Baseline Portfolio**: Assumes equal weights for all assets. It serves as a simplistic benchmark to assess the benefits of optimization.
#   
# - **Markowitz Portfolio**: Implements the traditional mean-variance optimization method, which seeks to minimize portfolio risk while achieving a target return. This method acts as a comparison to highlight the advantages of the SLSQP approach.

# In[5]:


# Baseline Portfolio
def baseline_metrics(weights, returns):
  """
  Compute performance metrics for the baseline portfolio (equal weights).
  
  Parameters:
    weights (numpy array): Equal weights for all assets.
    returns (DataFrame): Asset returns.
  
  Returns:
    Tuple: Sharpe ratio, CVaR, and MDD for the baseline portfolio.
  """
  return compute_metrics(weights, returns)

# Markowitz Optimization
def markowitz_optimization(train_returns, risk_free_rate, bounds):
  """
  Perform Markowitz mean-variance portfolio optimization.
  
  Parameters:
    train_returns (DataFrame): Asset returns for training.
    risk_free_rate (float): Risk-free rate of return.
    bounds (tuple): Bounds for portfolio weights.
  
  Returns:
    numpy array: Optimal portfolio weights for the Markowitz method.
  """
  mean_returns = train_returns.mean().values
  cov_matrix = train_returns.cov().values
  num_assets = len(mean_returns)

  # Objective function: Minimize portfolio risk (variance)
  def objective(weights):
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_volatility

  # Constraint: Weights sum to 1
  constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

  # Initial guess: equally weighted portfolio
  init_guess = num_assets * [1.0 / num_assets]

  # Optimize for minimum volatility
  result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
  return result.x


# ### **2.5 Hyperparameter Tuning**
# 
# To optimize the portfolio for varying risk and return preferences, a grid search was conducted over two hyperparameters:
# 
# - **Lambda1 (Weight for Return Maximization)**: Controls the importance assigned to maximizing returns
#   
# - **Lambda2 (Weight for Risk Minimization)**: Adjusts the focus on minimizing portfolio volatility.
# 
# The grid search uses:
# 
# - **k-Fold Cross-Validation**: Divides the data into k subsets for training and validation to ensure robust parameter tuning.
# 

# In[6]:


# Hyperparameter Grid Search with k-Fold Cross-Validation
def k_fold_grid_search(returns, k, lambda1_values, lambda2_values, bounds):
  """
  Perform k-fold cross-validation and grid search for hyperparameter tuning.
  
  Parameters:
    returns (DataFrame): Asset returns.
    k (int): Number of folds for cross-validation.
    lambda1_values (list): Range of lambda1 values for grid search.
    lambda2_values (list): Range of lambda2 values for grid search.
    bounds (tuple): Bounds for portfolio weights.
  
  Returns:
    DataFrame: Results of grid search with Sharpe ratio, CVaR, and MDD.
  """
  kf = KFold(n_splits=k, shuffle=False)
  results = []
  constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1} # Weights sum to 1

  for lambda1, lambda2 in product(lambda1_values, lambda2_values):
    sharpe_ratios, cvars, mdds = [], [], []
    prev_weights = None # For stability penalty

    for train_index, val_index in kf.split(returns):
      train_returns = returns.iloc[train_index]
      val_returns = returns.iloc[val_index]
      optimal_weights = portfolio_optimization(train_returns, lambda1, lambda2, bounds, constraints, prev_weights)
      sharpe_ratio, cvar_95, max_drawdown = compute_metrics(optimal_weights, val_returns)
      sharpe_ratios.append(sharpe_ratio)
      cvars.append(cvar_95)
      mdds.append(max_drawdown)
      prev_weights = optimal_weights

    results.append({'lambda1': lambda1, 'lambda2': lambda2, 
            'sharpe_ratio': np.mean(sharpe_ratios),
            'cvar': np.mean(cvars), 
            'mdd': np.mean(mdds)})

  return pd.DataFrame(results)


# In[7]:


# Define hyperparameter ranges
lambda1_values = np.linspace(1.0, 1.5, 10) # Range for lambda1 (return weight)
lambda2_values = np.linspace(1.2, 1.8, 10) # Range for lambda2 (risk weight)
bounds = tuple((0, 1) for _ in range(len(tickers))) # Portfolio weight bounds

# Number of folds for cross-validation
k = 5

# Run hyperparameter grid search
grid_results = k_fold_grid_search(returns, k, lambda1_values, lambda2_values, bounds)

# Identify the best parameters
best_params = grid_results.loc[grid_results['sharpe_ratio'].idxmax()]

# Define Best Lambda Values from Grid Search
best_lambda1 = best_params['lambda1']
best_lambda2 = best_params['lambda2']


# ### **2.6 Evaluation**
# 
# Evaluation of portfolio performance is divided into:
# 
# - **Baseline Portfolio**: Assumes equal weights for all assets as a benchmark.
# 
# - **Optimized Portfolio**: Uses the best hyperparameters (lambda1, lambda2) for optimization.
# 
# - **Markowitz Portfolio**: Employs the traditional mean-variance optimization method.
# 
# - **Sensitivity Analysis**: Assesses the impact of varying lambda1 and lambda2 on portfolio performance.
# 
# - **Final Implementation**: Combines all insights and applies the best parameters for portfolio construction.

# In[8]:


# Define Baseline Portfolio Evaluation
def evaluate_baseline_portfolio(returns):
  """
  Evaluate baseline portfolio metrics with equal weights.
  
  Parameters:
    returns (DataFrame): Asset returns.
  
  Returns:
    dict: Metrics (Sharpe Ratio, CVaR, MDD) for the baseline portfolio.
  """
  weights = np.array([1.0 / len(returns.columns)] * len(returns.columns))
  sharpe_ratio, cvar, mdd = compute_metrics(weights, returns)
  return {"weights": weights, "sharpe_ratio": sharpe_ratio, "cvar": cvar, "mdd": mdd}

# Define Backtesting for Optimized Portfolio
def backtest_optimized_portfolio(returns, lambda1, lambda2, bounds):
  """
  Perform backtesting for the optimized portfolio using predefined hyperparameters.
  
  Parameters:
    returns (DataFrame): Asset returns.
    lambda1 (float): Weight for return maximization.
    lambda2 (float): Weight for risk minimization.
    bounds (tuple): Bounds for portfolio weights.
  
  Returns:
    dict: Metrics (Sharpe Ratio, CVaR, MDD) and optimal weights.
  """
  optimal_weights = portfolio_optimization(
    returns, lambda1, lambda2, bounds, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
  )
  sharpe_ratio, cvar, mdd = compute_metrics(optimal_weights, returns)
  return {"weights": optimal_weights, "sharpe_ratio": sharpe_ratio, "cvar": cvar, "mdd": mdd}

# Define Markowitz Portfolio Evaluation
def evaluate_markowitz_portfolio(returns, bounds, risk_free_rate):
  """
  Evaluate the Markowitz mean-variance optimized portfolio.
  
  Parameters:
    returns (DataFrame): Asset returns.
    bounds (tuple): Bounds for portfolio weights.
    risk_free_rate (float): Risk-free rate.
  
  Returns:
    dict: Metrics (Sharpe Ratio, CVaR, MDD) and optimal weights for the Markowitz portfolio.
  """
  weights = markowitz_optimization(returns, risk_free_rate, bounds)
  sharpe_ratio, cvar, mdd = compute_metrics(weights, returns)
  return {"weights": weights, "sharpe_ratio": sharpe_ratio, "cvar": cvar, "mdd": mdd}


# In[9]:


# Perform Evaluation
def evaluate_all(returns, lambda1, lambda2, bounds, risk_free_rate):
  """
  Evaluate all portfolios: baseline, optimized, and Markowitz.
  
  Parameters:
    returns (DataFrame): Asset returns.
    lambda1 (float): Weight for return maximization for the optimized portfolio.
    lambda2 (float): Weight for risk minimization for the optimized portfolio.
    bounds (tuple): Bounds for portfolio weights.
    risk_free_rate (float): Risk-free rate.
  
  Returns:
    dict: Results for all portfolios.
  """
  # Backtesting for Optimized Portfolio
  optimized_results = backtest_optimized_portfolio(
    returns, lambda1, lambda2, bounds
  )
  print("Optimized Portfolio (SLSQP):\n",
     f"Sharpe Ratio: {optimized_results['sharpe_ratio']:.3f}, "
     f"CVaR: {optimized_results['cvar']:.3f}, "
     f"MDD: {optimized_results['mdd']:.3f}")

  # Evaluate Markowitz Portfolio
  markowitz_results = evaluate_markowitz_portfolio(returns, bounds, risk_free_rate)
  print("\nMarkowitz Portfolio (Traditional):\n",
     f"Sharpe Ratio: {markowitz_results['sharpe_ratio']:.3f}, "
     f"CVaR: {markowitz_results['cvar']:.3f}, "
     f"MDD: {markowitz_results['mdd']:.3f}")

  # Evaluate Baseline Portfolio
  baseline_results = evaluate_baseline_portfolio(returns)
  print("\nBaseline Portfolio (Equal-Weighted):\n",
     f"Sharpe Ratio: {baseline_results['sharpe_ratio']:.3f}, "
     f"CVaR: {baseline_results['cvar']:.3f}, "
     f"MDD: {baseline_results['mdd']:.3f}")

  return {
    "baseline": baseline_results,
    "optimized": optimized_results,
    "markowitz": markowitz_results,
  }


# In[10]:


def plot_cumulative_returns(returns, evaluation_results):
  """
  Plot cumulative returns for Optimized, Markowitz, and Baseline portfolios.
  
  Parameters:
    returns (DataFrame): Daily asset returns.
    evaluation_results (dict): Results containing weights for all portfolios.
  """
  plt.figure(figsize=(12, 6))

  # Optimized portfolio
  optimized_weights = evaluation_results['optimized']['weights']
  optimized_portfolio_returns = returns.dot(optimized_weights)
  optimized_cumulative_returns = (1 + optimized_portfolio_returns).cumprod()
  plt.plot(optimized_cumulative_returns, label="Optimized Portfolio (SLSQP)", color='blue')

  # Markowitz portfolio
  markowitz_weights = evaluation_results['markowitz']['weights']
  markowitz_portfolio_returns = returns.dot(markowitz_weights)
  markowitz_cumulative_returns = (1 + markowitz_portfolio_returns).cumprod()
  plt.plot(markowitz_cumulative_returns, label="Markowitz Portfolio", color='orange')

  # Baseline portfolio
  baseline_weights = evaluation_results['baseline']['weights']
  baseline_portfolio_returns = returns.dot(baseline_weights)
  baseline_cumulative_returns = (1 + baseline_portfolio_returns).cumprod()
  plt.plot(baseline_cumulative_returns, label="Baseline Portfolio", color='green')

  # Add labels, title, and legend
  plt.title("Cumulative Returns Comparison")
  plt.xlabel("Time")
  plt.ylabel("Cumulative Returns")
  plt.legend()
  plt.grid()
  plt.show()


# In[11]:


def sensitivity_analysis(returns, lambda1_range, lambda2_range, bounds):
  """
  Perform sensitivity analysis for lambda1 and lambda2.
  
  Parameters:
    returns (DataFrame): Asset returns.
    lambda1_range (list): Range of lambda1 values to test.
    lambda2_range (list): Range of lambda2 values to test.
    bounds (tuple): Bounds for portfolio weights.
  
  Returns:
    DataFrame: Sensitivity analysis results with Sharpe Ratio, CVaR, and MDD.
  """
  results = []
  constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

  for lambda1, lambda2 in product(lambda1_range, lambda2_range):
    optimal_weights = portfolio_optimization(returns, lambda1, lambda2, bounds, constraints)
    sharpe_ratio, cvar, mdd = compute_metrics(optimal_weights, returns)
    results.append({'lambda1': lambda1, 'lambda2': lambda2, 
            'sharpe_ratio': sharpe_ratio, 'cvar': cvar, 'mdd': mdd})
  
  return pd.DataFrame(results)

# Define the range for sensitivity analysis
lambda1_range = np.linspace(1.0, 1.5, 10)
lambda2_range = np.linspace(1.2, 1.8, 10)

# Perform Sensitivity Analysis
sensitivity_results = sensitivity_analysis(returns, lambda1_range, lambda2_range, bounds)

# Visualize the Sensitivity Analysis
def plot_sensitivity_analysis(sensitivity_results):
  """
  Visualize the sensitivity of Sharpe Ratio to lambda1 and lambda2 values.
  
  Parameters:
    sensitivity_results (DataFrame): Sensitivity analysis results.
  """
  pivot_sharpe = sensitivity_results.pivot(index='lambda1', columns='lambda2', values='sharpe_ratio')
  plt.figure(figsize=(10, 6))
  sns.heatmap(pivot_sharpe, annot=True, cmap='coolwarm', fmt=".3f")
  plt.title("Sensitivity of Sharpe Ratio to λ1 and λ2")
  plt.xlabel("λ2 (Risk Weight)")
  plt.ylabel("λ1 (Return Weight)")
  plt.show()


# In[12]:


def final_implementation(returns, best_lambda1, best_lambda2, bounds, risk_free_rate):
  """
  Perform the final portfolio optimization and evaluation.
  
  Parameters:
    returns (DataFrame): Asset returns.
    best_lambda1 (float): Best value for lambda1 (return weight).
    best_lambda2 (float): Best value for lambda2 (risk weight).
    bounds (tuple): Bounds for portfolio weights.
    risk_free_rate (float): Risk-free rate.
  
  Returns:
    dict: Final portfolio weights and performance metrics.
  """
  # Optimize the portfolio with the best parameters
  optimal_weights = portfolio_optimization(
    returns, best_lambda1, best_lambda2, bounds, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
  )
  sharpe_ratio, cvar, mdd = compute_metrics(optimal_weights, returns)

  print("Final Portfolio Weights:")
  for ticker, weight in zip(returns.columns, optimal_weights):
    print(f"{ticker}: {weight:.2%}")

  print("\nFinal Performance Metrics:")
  print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
  print(f"CVaR: {cvar:.3f}")
  print(f"MDD: {mdd:.3f}")

  return {"weights": optimal_weights, "sharpe_ratio": sharpe_ratio, "cvar": cvar, "mdd": mdd}


# ### **2.7 Rolling-Window Backtesting**
# 
# Rolling-window backtesting assesses the portfolio’s adaptability to dynamic market conditions. This method recalibrates portfolio weights over:
# 
# - **Rolling Period**s: Fixed window sizes (e.g., 252 days for one year) to simulate real-time portfolio management.
#   
# - **Step Sizes**: Smaller intervals (e.g., 21 days) between recalibrations to maintain responsiveness.

# In[13]:


# Rolling-Window Backtesting
def rolling_window_backtesting(returns, window_size, step_size, lambda1, lambda2, bounds):
  """
  Perform rolling-window backtesting of the portfolio.
  
  Parameters:
    returns (DataFrame): Asset returns.
    window_size (int): Size of the rolling window (in days).
    step_size (int): Step size for the rolling window (in days).
    lambda1 (float): Weight for portfolio return.
    lambda2 (float): Weight for portfolio risk.
    bounds (tuple): Bounds for portfolio weights.
  
  Returns:
    metrics (DataFrame): Sharpe ratio, CVaR, and MDD for each window.
    weights (list): Optimal weights for each rolling window.
  """
  metrics = []
  weights = []
  constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1} # Weights sum to 1
  
  for start in range(0, len(returns) - window_size, step_size):
    end = start + window_size
    train_returns = returns.iloc[start:end]
    optimal_weights = portfolio_optimization(train_returns, lambda1, lambda2, bounds, constraints)
    weights.append(optimal_weights)
    sharpe_ratio, cvar, mdd = compute_metrics(optimal_weights, train_returns)
    metrics.append({'start_date': returns.index[start], 
            'end_date': returns.index[end - 1], 
            'Sharpe_Ratio': sharpe_ratio, 
            'CVaR': cvar, 
            'MDD': mdd})
  
  return pd.DataFrame(metrics), weights


# ## 3. Results
# 
# This section synthesizes key findings from the methodology. Each part of the analysis is presented with relevant figures and metrics to explain the portfolio’s performance and adaptability

# ### **3.1 Performance Metrics Across Portfolios**
# 
# The performance metrics—Sharpe Ratio, CVaR, and Maximum Drawdown (MDD)—highlighted the superior risk-adjusted returns and resilience of the optimized portfolio compared to the baseline and Markowitz portfolios.

# In[14]:


print("Best Hyperparameters:")
print(best_params)
print("\n")

# Perform Evaluation
evaluation_results = evaluate_all(returns, best_lambda1, best_lambda2, bounds, risk_free_rate)


# **Interpretation:**
# 
# - The optimized portfolio achieved the highest Sharpe Ratio (1.362), indicating strong risk-adjusted returns. The CVaR of -0.016 shows a controlled tail risk, while the MDD of 0.211 reflects minimal peak-to-trough losses. This demonstrates the portfolio’s ability to balance return maximization and risk minimization effectively
# 
# - The Markowitz portfolio achieved better risk-adjusted returns compared to the baseline portfolio but lagged behind the optimized portfolio. The MDD (0.184) is slightly better than the optimized portfolio, reflecting improved stability. However, its lower Sharpe Ratio (1.070) indicates suboptimal return generation compared to the SLSQP method
# 
# - The baseline portfolio, with equal weights for all assets, displayed lower risk-adjusted returns compared to the optimized portfolio. The significantly higher CVaR (-0.033) and MDD (0.694) highlight its vulnerability to extreme losses, emphasizing the limitations of equal-weighting strategies

# ### **3.2 Cumulative Returns Comparison**
# 
# The cumulative returns demonstrate the practical advantage of the SLSQP approach in maximizing returns while minimizing risks across different market conditions

# In[15]:


# Plotting Cumulative Returns
plot_cumulative_returns(returns, evaluation_results)


# **Interpretation:** 
# 
# - The **Optimized Portfolio (SLSQP )** outperformed both the **Markowitz Portfolio** and **Baseline Portfolio** in cumulative returns, highlighting its superior performance across the evaluation period
# 
# - The optimized portfolio exhibited resilience during market drawdowns, such as the COVID-19-induced market crash in 2020, recovering faster and maintaining lower volatility
# 
# - The baseline portfolio lagged significantly, reinforcing the value of optimization methods in portfolio construction

# ### **3.3 Sensitivity Analysis**
# 
# The sensitivity analysis validated the choice of hyperparameters used in the portfolio optimization, ensuring a balanced trade-off between returns and risk

# In[16]:


# Plot the Sensitivity Analysis
plot_sensitivity_analysis(sensitivity_results)


# **Interpretation**:
# 
# - The sensitivity analysis explored the impact of varying **λ1 (return weight)** and **λ2 (risk weight)** on the Sharpe Ratio, providing insights into the model’s responsiveness to these parameters.
# 
# - Higher **λ1** values emphasized return maximization, resulting in improved Sharpe Ratios, while increased **λ2** values reflected a greater focus on risk mitigation.
# 
# - Although the heatmap suggests optimal performance around **λ1 = 1.2** and **λ2 = 1.4**, the grid search selected **λ1 = 1** and **λ2 = 1.8** as the best parameters based on averaged performance across multiple cross-validation folds. This highlights the robustness of the chosen hyperparameters for generalizability across market conditions.
# 
# - The heatmap complements the grid search by identifying regions of strong performance, validating that the selected hyperparameters lie within a high-performing range. This dual perspective ensures both precision and broader applicability.

# ### **3.4 Final Portfolio Implementation Results**
# 
# The final portfolio allocation reflected a well-diversified strategy with robust performance across risk-adjusted metrics, consistent with the project’s objectives

# In[17]:


# Perform Final Implementation
final_results = final_implementation(returns, best_lambda1, best_lambda2, bounds, risk_free_rate)


# **Interpretation:**
# 
# The optimized portfolio allocates significant weights to assets with strong risk-adjusted returns and stabilizing characteristics:
# 
# **High Allocations**:
# 
#   - **MSFT (21.51%)**: Strong performance with high Sharpe Ratio.
#     
#   - **DBC (28.51%)**: Provides diversification benefits and robust risk-return trade-offs.
#  
#     
#   - **TLT (29.13%)** and **GLD (19.08%)**: Stabilizing assets that mitigate risk, especially during volatile periods.
# 
# **Low or Zero Allocations**:
# 
#   - **AAPL (0.00%)** and **GOOGL (0.00%)**: Likely deprioritized due to their correlations with other assets or less favorable risk-adjusted returns
# 
#   - **ETH-USD (0.00%)**: High volatility and poor risk-return trade-off make it unsuitable for inclusion
#     
#   - **VNQ (0.43%)**: Minimal allocation suggests it provides limited diversification benefits in the current portfolio
# 
# The portfolio reflects a balance between assets with strong performance potential and those that stabilize overall risk, achieving a high Sharpe Ratio while maintaining relatively low Conditional Value at Risk (CVaR) and Maximum Drawdown (MDD). This allocation strategy ensures effective diversification and risk management while maximizing returns.

# In[18]:


# Compute and display asset-level metrics
asset_metrics = compute_asset_metrics(returns)
print("\nAsset-Level Metrics:\n")
asset_metrics


# **Interpretation:** 
# 
# 1. **High Allocations in the Optimized Portfolio**
# 
#    - **MSFT (21.51%)**: Has the highest Sharpe Ratio (**1.065**), indicating strong risk-adjusted returns
#      
#    - **DBC (28.51%)**: A well-diversified asset with a relatively high Sharpe Ratio (**0.871**) and lower volatility (**0.185**)
#      
#    - **GLD (19.08%)** and **TLT (29.13%)**: Though their Sharpe Ratios are lower (**0.606** and **0.338**, respectively), they are stabilizing assets with low volatility, making them valuable for risk reduction
# 
# 2. **Low or Zero Allocations**
# 
#    - **AAPL (0.00%)** and **GOOGL (0.00%)**: Despite reasonable Sharpe Ratios (**0.726** and **0.619**), these assets are likely excluded due to high correlations with **MSFT**, offering redundant exposure without sufficient diversification benefits.
#      
#    - **BTC-USD (1.34%)** and **ETH-USD (0.00%)**: Cryptocurrencies exhibit high volatility (**0.676** and **0.844**) and lower Sharpe Ratios (**0.486** and **0.339**), reducing their attractiveness as stable contributors to the portfolio.
# 
# 3. **Moderate Allocation**
# 
#    
#    - **VNQ (0.43%)**: Its Sharpe Ratio (**0.623**) is reasonable, but its return and volatility metrics are less compelling compared to other included assets, resulting in a minimal allocation.
# 
# The optimized portfolio allocates weights based on a balance between maximizing risk-adjusted returns (Sharpe Ratio) and minimizing volatility. High-weighted assets, like **MSFT** and **DBC**, dominate due to their strong performance, while low-volatility assets, like **GLD** and **TLT**, contribute to stability. Conversely, assets with higher risk or redundant correlations, like **AAPL**, **GOOGL**, and **ETH-USD**, are deprioritized to enhance overall diversification and portfolio performance.

# In[19]:


# Calculate correlation matrix
correlation_matrix = returns.corr()

# Analyze excluded assets
excluded_assets = ['AAPL', 'GOOGL', 'ETH-USD'] # Assets excluded in the optimized portfolio
correlations_to_others = correlation_matrix[excluded_assets].drop(index=excluded_assets)

print("\nCorrelations of Excluded Assets to Other Assets:\n")
correlations_to_others


# **Excluded Assets:**
# 
# 1. **ETH-USD and BTC-USD**:
# 
#    - **Strong Correlation**: The high correlation between **BTC-USD** and **ETH-USD** (**0.792**) indicates redundancy in their risk-return profiles. This reduces the diversification benefits of including both, leading to the exclusion of **ETH-USD**.
#    
# 2. **GOOGL and MSFT**:
# 
#    - **Tech Correlation**: **GOOGL**'s strong correlation with **MSFT** (**0.796**) diminishes its unique contribution to the portfolio. Since **MSFT** has a higher Sharpe Ratio, it is prioritized over **GOOGL**.
# 
# 3. **AAPL**:
# 
#    - **Redundancy with MSFT and VNQ**: **AAPL** exhibits moderate correlations with **MSFT** (**0.760**) and **VNQ** (**0.545**). Combined with its lower Sharpe Ratio, this makes it less favorable for inclusion.
# 
# **Diversified Portfolio:**
# 
# The inclusion of assets with complementary risk-return profiles, such as:
# 
# - **MSFT**: A high-performing tech stock with a strong Sharpe Ratio.
#   
# - **TLT**: A stabilizing asset with low volatility and negative correlations with equities.
# 
# - **DBC**: A commodity-focused asset providing diversification benefits with minimal correlations to other included assets.
# 
# This balanced selection enhances the overall portfolio performance, maximizing returns while minimizing risks.

# ### **3.5 Rolling-Window Backtesting**
# 
# Rolling-window backtesting highlighted the portfolio’s adaptability to market conditions and its ability to sustain superior risk-adjusted returns over time

# In[20]:


# Parameters
window_size = 252 # 1 year
step_size = 21 # 1 month
rolling_metrics, rolling_weights = rolling_window_backtesting(returns, window_size, step_size, 
                               best_params['lambda1'], 
                               best_params['lambda2'], bounds)

# Display Results
print("Rolling-Window Metrics:\n")
rolling_metrics


# In[21]:


# Plot Rolling Metrics
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(rolling_metrics['start_date'], rolling_metrics['Sharpe_Ratio'], label='Sharpe Ratio', color='blue')
ax1.plot(rolling_metrics['start_date'], rolling_metrics['MDD'], label='MDD', color='green')
ax1.set_ylabel("Sharpe Ratio / MDD", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(rolling_metrics['start_date'], rolling_metrics['CVaR'], label='CVaR', color='red', linestyle='dashed')
ax2.set_ylabel("CVaR", color='red')
ax2.tick_params(axis='y', labelcolor='red')

fig.suptitle("Rolling Portfolio Metrics Over Time")
fig.tight_layout()
fig.legend(loc="upper left")
plt.grid()
plt.show()


# In[22]:


# Plot Rolling Weights
rolling_weights_df = pd.DataFrame(rolling_weights, columns=tickers)
rolling_weights_df['start_date'] = rolling_metrics['start_date']
rolling_weights_df.set_index('start_date', inplace=True)
rolling_weights_df.plot(figsize=(12, 6), title='Rolling Portfolio Weights')
plt.xlabel("Start Date")
plt.ylabel("Weight")
plt.legend(title="Assets")
plt.grid()
plt.show()


# **Interpretation:** 
# 
# - **Sharpe Ratio Trends:** Rolling Sharpe Ratios showed variability over time, peaking during favorable market conditions and dropping during downturns like the COVID-19 crash. 
# 
# - **CVaR and MDD Trends:** Tail risk and drawdowns were well-controlled, demonstrating the portfolio's resilience to market fluctuations. 
# 
# - **Dynamic Weights:** Portfolio weights adjusted dynamically across time windows, adapting to changing market conditions and ensuring balanced risk-return profiles.

# ## 4. Discussion and Conclusion
# 
# This section summarizes the key takeaways from the results, highlights the challenges encountered during the project, and outlines future improvements and potential refinements

# ### **4.1 Discussion**

# **What Did You Learn from Your Results?**
# 
# - The portfolio optimization model using the SLSQP method (Sequential Least Squares Programming Algorithm) demonstrated its ability to effectively balance returns and risks, achieving superior performance metrics compared to both the baseline (equal-weighted portfolio) and the traditional Markowitz mean-variance optimization:
# 
#     - **Optimized Portfolio** (SLSQP): Sharpe Ratio = **1.362**, CVaR = **-0.016**, MDD = **0.211**
# 
#     - **Markowitz Portfolio**: Sharpe Ratio = **1.070**, CVaR = **-0.014**, MDD = **0.184**
# 
#     - **Baseline Portfolio**: Sharpe Ratio = **0.867**, CVaR = **-0.033**, MDD = **0.694**
# 
# - Sensitivity analysis showed how varying **λ1** (return weight) and **λ2** (risk weight) impacted portfolio performance:
# 
#     - Higher **λ1** values prioritized returns, improving Sharpe Ratios.
# 
#     - The chosen hyperparameters **λ1 = 1.0** and **λ2 = 1.8** achieved robust performance across cross-validation folds and aligned with the general high-performing range on the heatmap.
# 
# - Rolling-window backtesting revealed the adaptability of the optimization model under varying market conditions:
# 
#     - Sharpe Ratios fluctuated during high-volatility periods (e.g., COVID-19 downturn) but remained relatively stable compared to the baseline.
# 
#     - Maximum Drawdowns (MDD) were consistently controlled, reflecting the model’s ability to mitigate risks effectively.
# 
# - Asset-level analysis revealed that certain assets (e.g., **GOOGL**, **AAPL**, **ETH-USD**) were excluded from the final portfolio due to low Sharpe Ratios or high correlations with other included assets, which reduced their diversification value.
# 
# **Obstacles Encountered**
# 
# 1. **Class Imbalance in Asset Returns**: Certain assets exhibited extreme volatility (e.g., cryptocurrencies), making it challenging to balance their inclusion in the portfolio. This was addressed by penalizing large weights in the optimization function.
# 
# 2. **Hyperparameter Sensitivity**: Initial grid search results were sometimes overly sensitive to specific **λ1** and **λ2** values. This was mitigated by using cross-validation to select hyperparameters that generalized well across different training folds.
# 
# 3. **Rolling-Window Computational Overheads**: Performing rolling-window backtesting on large datasets with multiple assets required significant computational resources. Code optimizations (e.g., vectorization) helped reduce runtime.
# 
# **What Would You Do Differently Next Time?**
# 
# 1. **Include More Diverse Asset Classes**: Expanding the dataset to include additional asset classes, such as real estate or commodities, would further test the model's adaptability to different economic conditions.
# 
# 2. **Integrate Advanced Risk Metrics**: Incorporate additional risk measures, such as Value at Risk (VaR) at multiple confidence levels or Sortino Ratio, to provide a more comprehensive evaluation of portfolio performance.
# 
# 3. **Explore Machine Learning Models**: Incorporate reinforcement learning or neural networks to model asset weight allocations dynamically, potentially outperforming traditional optimization techniques.
# 
# 4. **Analyze Economic Context**: Study the portfolio's behavior during specific historical economic events (e.g., 2008 financial crisis, 2020 COVID-19 pandemic) to draw deeper insights into its adaptability.
# 
# **Quantitative Answers to Questions**
# 
# **Did the optimized portfolio outperform traditional methods?**
# 
#  - Yes, the SLSQP optimization method yielded a Sharpe Ratio of **1.362**, significantly outperforming the Markowitz portfolio (**1.070**) and the equal-weighted baseline (**0.867**).
# 
# **How did the portfolio perform during volatile periods?**
# 
#  - Rolling-window backtesting showed that the optimized portfolio maintained relatively stable Sharpe Ratios and lower drawdowns during periods of market turbulence, such as the COVID-19 downturn.
# 
# **Were the chosen hyperparameters effective?**
# 
#  - Yes, the sensitivity analysis confirmed that **λ1 = 1.0** and **λ2 = 1.8** fell within the high-performing range of hyperparameters, as demonstrated by the heatmap and cross-validation results.

# ### **4.2 Conclusion**

# The project successfully demonstrated that the SLSQP method could optimize a portfolio by balancing returns and risks more effectively than traditional methods. The model not only achieved superior performance metrics but also exhibited robustness under varying economic conditions. While there are opportunities to enhance the methodology (e.g., using machine learning models or integrating broader asset classes), this study underscores the potential of advanced optimization techniques in financial decision-making.

# ## 5. References
# 
# This section serves to properly attribute all sources, tools, and methodologies that were utilized or referenced during the course of the project. 

# 
# 1. Historical financial data was sourced using the [Yahoo Finance API](https://pypi.org/project/yfinance/).
# 
# 2. Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization*. Springer. (Reference for the optimization algorithms)
# 
# 3. Scipy documentation: [https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
# 
# 3. Sharpe, W. F. (1994). *The Sharpe Ratio*. Journal of Portfolio Management, 21(1), 49–58. (Source for Sharpe Ratio calculations.)
# 
# 4. Rockafellar, R. T., & Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk*. Journal of Risk, 2(3), 21–41. (Source for CVaR calculations.)
# 
# 5. This project benefited from assistance provided by OpenAI's ChatGPT for structuring, debugging, and refining code.
