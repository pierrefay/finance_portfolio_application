import math
from datetime import time, datetime, timedelta
from decimal import Decimal, ROUND_DOWN
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import statsmodels as sm

DAYS_IN_YEAR = 252
DAYS_IN_MONTH = 252/12

def number_of_days(df):
    return df.shape[0] - df.isna().sum()

def total_returns(df):
    return np.prod(df+1) - 1

def return_per_day(df):
    nbr_of_actual_days = number_of_days(df)
    return_per_day = ((df+1).prod() ** (1 / nbr_of_actual_days)) - 1
    return return_per_day

def periodized_returns(df, nbr_days=DAYS_IN_MONTH):
    return ((return_per_day(df) +1) ** nbr_days)-1

def volatility(df):
    return df.std()

def periodized_volatility(df, nbr_days=DAYS_IN_MONTH):
    return df.std()*np.sqrt(nbr_days)

def sharpe_ratio(df):
    return periodized_returns(df, nbr_days=DAYS_IN_YEAR)/periodized_volatility(df,  nbr_days=DAYS_IN_YEAR)

def semi_deviation(df):
    return df[df<0].std(ddof=0)

def max_drawdown(df):
    #on prend le max a chaque temps, on retire le prix => on prend la valeur du max drawdown
    wealth_index = (1 + df).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    max_drawdowns = drawdowns.min()
    return max_drawdowns

#Historical VaR
def value_at_risk(df, level=5):
    if isinstance(df, pd.DataFrame):
        return df.aggregate(value_at_risk, level=level)
    elif isinstance(df, pd.Series):
        return - np.percentile(df.dropna(), level)
    else:
        raise TypeError('df must be a valid Dataframe or Series')

    # calculer le max loss a une certaine probabilité
    # a calculer via cornish-fischer car plus robuste (pas besoin d'une distribution de proba speciale ex: normale)
    return True

# portfolio returns
def portfolio_return(weights, returns):
    """
    Weights to returns
    """
    return weights.T @ returns

def portfolio_stats(df_returns):
    test = pd.DataFrame()
    test['active_days'] = number_of_days(df_returns)
    test['total_returns'] = total_returns(df_returns)
    test['daily_returns'] = periodized_returns(df_returns, nbr_days=1)
    test['daily_volatility'] = volatility(df_returns)
    test['monthly_returns'] = periodized_returns(df_returns, nbr_days=DAYS_IN_MONTH)
    test['monthly_volatility'] = periodized_volatility(df_returns, nbr_days=DAYS_IN_MONTH)
    test['yearly_returns'] = periodized_returns(df_returns, nbr_days=DAYS_IN_YEAR)
    test['yearly_volatility'] = periodized_volatility(df_returns, nbr_days=DAYS_IN_YEAR)
    test['sharpe_ratio'] = sharpe_ratio(df_returns)
    test['max_drawdown'] = max_drawdown(df_returns)
    test['VaR'] = value_at_risk(df_returns)
    return test

def portfolio_volatility(weights, cov):
    return np.sqrt(weights.T @ cov @ weights)

def portfolio_sharpe_ratio(weights, returns, cov, riskfreerate):
        r = portfolio_return(weights, returns)
        v = portfolio_volatility(weights, cov)
        s = (r - riskfreerate)/v
        return s

def portfolio_max_renta(weights, returns):
    r = portfolio_return(weights, returns)
    return r

def portfolio_negative_max_renta(weights, returns):
    return -portfolio_return(weights, returns)

def portfolio_negative_sharpe_ratio(weights, returns, cov, riskfreerate):
    return -portfolio_sharpe_ratio(weights, returns, cov, riskfreerate)

def optimal_weigths(number_of_points, returns, cov):
    target_returns = np.linspace(returns.min(), returns.max(), number_of_points)
    weights = [minimize_vol(tr, returns, cov) for tr in target_returns]
    return weights

def plot_efficient_frontier(number_of_points, expectedreturns, cov, riskfreerate=0, show_cml=False, show_ew=False, show_gmv=False):
    report_weights = pd.DataFrame(index=['GMV','MSR','EW'], columns=cov.columns)
    weights = optimal_weigths(number_of_points, expectedreturns, cov)
    preturns = [ portfolio_return(w, expectedreturns) for w in weights]
    pvolatility = [ portfolio_volatility(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": preturns, "Volatility": pvolatility})
    ax = ef.plot.line(x="Volatility", y="Returns", style=".-", label="Efficient Frontier")
    if show_gmv:
        gmv_weight = global_minimum_variance(cov)
        report_weights.loc['GMV'] = gmv_weight
        gmv_returns = portfolio_return(gmv_weight, expectedreturns)
        gmv_volatility = portfolio_volatility(gmv_weight, cov)
        ax.plot([gmv_volatility], [gmv_returns], color="red", marker="o", markersize=10, label="Global Minimum Variance (GMV) portfolio")

    if show_ew:
        number_of_assets = expectedreturns.shape[0]
        ew_weight = np.repeat(1/number_of_assets, number_of_assets)
        report_weights.loc['EW'] = ew_weight
        ew_returns = portfolio_return(ew_weight, expectedreturns)
        ew_volatility = portfolio_volatility(ew_weight, cov)
        ax.plot([ew_volatility], [ew_returns], color="goldenrod", marker="o", markersize=10,  label="Equality Weighted (EW) portfolio (naive diversification)")

    if show_cml:
        max_sr_weight = max_sharperatio(expectedreturns, cov, riskfreerate=riskfreerate)
        report_weights.loc['MSR'] = max_sr_weight
        max_sr_returns = portfolio_return(max_sr_weight, expectedreturns)
        max_sr_volatility = portfolio_volatility(max_sr_weight, cov)

       # ax.set_xlim(left=0)
        cml_x = [0, max_sr_volatility]
        cml_y = [riskfreerate, max_sr_returns]

        ax.plot(cml_x, cml_y, color="green", marker="o", linestyle="dashed", label="Capital Market Line (CML) (Max Sharp Ratio)")
    ax.legend()
    print("\n\n poids par asset par stragégie")
    print(report_weights)
    plt.show()
    return report_weights

def minimize_vol(target_return, returns, cov):
    number_of_assets = returns.shape[0]
    init_guess = np.repeat(1/number_of_assets, number_of_assets)
    bounds = ((0.0, 1.0),)*number_of_assets
    return_is_target = {
        'type': 'eq',
        'args': (returns,),
        'fun': lambda weights, expected_returns: target_return - portfolio_return(weights, returns)
    }
    weight_sum_to_one = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    results = minimize(portfolio_volatility,
                       init_guess,
                       args=(cov,),
                       method="SLSQP", #quadratic optimizer
                       options={"maxiter":10, 'disp': False},
                       constraints=(return_is_target, weight_sum_to_one),
                       bounds=bounds
                       )
    return results.x

def max_sharperatio(returns, cov, riskfreerate=0):
    number_of_assets = returns.shape[0]
    init_guess = np.repeat(1/number_of_assets, number_of_assets)
    bounds = ((0.0, 1.0),)*number_of_assets
    weight_sum_to_one = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    results = minimize(portfolio_negative_sharpe_ratio,
                       init_guess,
                       args=(returns, cov, riskfreerate),
                       method="SLSQP", #quadratic optimizer
                       options={"maxiter":10, 'disp': False},
                       constraints=(weight_sum_to_one),
                       bounds=bounds
                       )
    return results.x

def min_sharperatio(returns, cov, riskfreerate=0):
    number_of_assets = returns.shape[0]
    init_guess = np.repeat(1/number_of_assets, number_of_assets)
    bounds = ((0.0, 1.0),)*number_of_assets
    weight_sum_to_one = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    results = minimize(portfolio_sharpe_ratio,
                       init_guess,
                       args=(returns, cov, riskfreerate),
                       method="SLSQP", #quadratic optimizer
                       options={"maxiter":10, 'disp': False},
                       constraints=(weight_sum_to_one),
                       bounds=bounds
                       )
    return results.x

def max_renta(returns, min_assets=3):
    number_of_assets = returns.shape[0]

    init_guess_0 = np.repeat(0, number_of_assets-min_assets)
    init_guess_1 = np.repeat(1/min_assets, min_assets)
    init_guess = np.concatenate((init_guess_1, init_guess_0))

    bounds = ((0.0, 1.0),)*number_of_assets
    weight_sum_to_one = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    minimum_assets_number = {
        'type': 'eq',
        'fun': lambda weights: np.count_nonzero(weights) - min_assets
    }
    results = minimize(portfolio_negative_max_renta,
                       init_guess,
                       args=(returns),
                       method="SLSQP", #quadratic optimizer
                       options={"maxiter":10, 'disp': False},
                       constraints=(weight_sum_to_one),
                       bounds=bounds
                       )
    if results.success:
        print(results)
        exit()
        return results.x
    else:
        print(results)
        exit()

def global_minimum_variance(cov):
    number_of_assets = cov.shape[0]
    init_guess = np.repeat(1 / number_of_assets, number_of_assets)
    bounds = ((0.0, 1.0),) * number_of_assets
    weight_sum_to_one = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    results = minimize(portfolio_volatility,
                       init_guess,
                       args=(cov),
                       method="SLSQP",  # quadratic optimizer
                       options={"maxiter":10, 'disp': False},
                       constraints=(weight_sum_to_one),
                       bounds=bounds
                       )
    return results.x


def geometric_brownian_motion(n_period=10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_period=12, s_0=100.0):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1 / steps_per_period
    n_steps = int(n_period * steps_per_period) + 1
    rets_plus_1 = np.random.normal(loc=mu * dt + 1, scale=sigma * np.sqrt(dt), size=(n_steps, n_scenarios))
    # or better ...
    # rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    prices = s_0 * pd.DataFrame(rets_plus_1).cumprod()
    return prices

def capm(r_port,r_market):
    capm_model = sm.OLS(r_port,r_market).fit()
    capm_model.summary()

def weight_ew(df_returns, **kwargs):
    count = float(Decimal(1/df_returns.shape[1]).quantize(Decimal('.00001'), rounding=ROUND_DOWN))
    df = pd.DataFrame(index=df_returns.index, columns=df_returns.columns)
    df.fillna(count, inplace=True)
    return df

def sample_cov(df_returns, **kwargs):
    return df_returns.cov()

def weight_gmv(df_returns, cov_estimator=sample_cov, **kwargs):
    testdf = df_returns.dropna(axis=1)
    cov = cov_estimator(testdf, **kwargs)
    gmv_w = global_minimum_variance(cov)
    date = df_returns.index.values[-1:]
    test = pd.DataFrame(index=date, columns=df_returns.columns.values)
    for col in testdf.columns.values:
       test.loc[date, col] = gmv_w[list(testdf.columns.values).index(col)]
    return test

def weight_msr(df_returns, cov_estimator=sample_cov, **kwargs):
    testdf = df_returns.dropna(axis=1)
    cov = cov_estimator(testdf, **kwargs)
    testdf_periodized = periodized_returns(testdf, nbr_days=1)
    msr_w = max_sharperatio(testdf_periodized, cov)
    date = df_returns.index.values[-1:]
    test_buy = pd.DataFrame(index=date, columns=df_returns.columns.values)
    for col in testdf.columns.values:
       test_buy.loc[date, col] = msr_w[list(testdf.columns.values).index(col)]
    return test_buy


def weight_imsr(df_returns, cov_estimator=sample_cov, **kwargs):
    testdf = df_returns.dropna(axis=1)
    cov = cov_estimator(testdf, **kwargs)
    testdf_periodized = periodized_returns(testdf, nbr_days=1)
    msr_w = min_sharperatio(testdf_periodized, cov)
    date = df_returns.index.values[-1:]
    test_buy = pd.DataFrame(index=date, columns=df_returns.columns.values)
    for col in testdf.columns.values:
       test_buy.loc[date, col] = msr_w[list(testdf.columns.values).index(col)]
    return test_buy

def set_state(row):
    if row['buy'] > 0 and row['sell'] == 0:
            return row['buy']
    else:
        if row['sell'] == 1:
            return 0
        else:
            return None