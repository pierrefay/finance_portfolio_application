# Etape 1- recupération des données de Yahoo Finance et création de la matrice des returns
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tools

#liste des symbole qu'on veut recupérer
symbols = ['MSFT','AAPL','TSLA','BTC-USD','^GSPC','PM','KO','AAL','AXP']
returns_matrix = pd.DataFrame()
price_matrix = pd.DataFrame()

#on boucle sur tous ce qu'on veut récupérer pour mettre dans notre portefeuille
for symbol in symbols:
    #on recupere l'historique
    ticker = yf.Ticker(symbol)
    symboldf = ticker.history(interval='1d', start='2015-01-01')
    # symboldf = ticker.history(interval='1d', start='2020-01-01', end='2021-04-01')

    #on calcule le returns
    # on supprime la premiere ligne qui est un NaN
    # on rename la colonne "returns" en AAPL par exemple pour la stocker directement dans la matrice avec le bon nom
    symboldf['returns'] = symboldf['Close'].pct_change()
    symboldf.dropna(inplace=True)
    symboldf.rename(columns={'returns':symbol}, inplace=True)

    #on stocke le return dans la matrice
    returns_matrix[symbol] = symboldf[symbol]

    #on stocke le prix
    #on rename la colonne AAPL par exemple en "returns" puis Close par APPL pour pouvoir le stocker avec le bon nom dans la matrice
    symboldf.rename(columns={symbol: 'returns', 'Close':symbol}, inplace=True)
    price_matrix[symbol] = symboldf[symbol]

#on sauvegarde en csv nos tableaux prix de cloture et returns
returns_matrix.to_csv('returns_matrix.csv')
price_matrix.to_csv('price_matrix.csv')

#on affiche les prix
# price_matrix.plot()
# plt.yscale('log') #echelle logarithmique pour l'axe Y
# plt.show()

# Etape 2- Description des données et calcul maxdrawdown, sharpe ratio, volatilité  et returns rapporté sur 1 et 30 jours et VaR
# voir les détails de chaque calculs
stats = tools.portfolio_stats(returns_matrix)
print('\n\n description des différents indices')
print(stats)


#frontiere efficiente
# liens volatilité / risque
# "magie" de la diversification sur la volatilité
assets = ['MSFT','AAPL','TSLA','BTC-USD','PM','KO','AAL','AXP']
number_of_points = 20
dfreturns = returns_matrix[assets]
returns = stats['total_returns'][assets]

cov = dfreturns.cov()
weights = tools.plot_efficient_frontier(number_of_points, returns, cov, riskfreerate=0, show_cml=True, show_ew=True, show_gmv=True)

# Etape 3- Backtest des différentes stratégies
print('\n\nreturns par stratégie:')
results = pd.DataFrame(columns=weights.index, index=returns.index)
print((returns*weights).sum(axis=1))

returns_matrix['my_portfolio'] =0
for asset in assets:
    returns_matrix['my_portfolio'] = returns_matrix['my_portfolio'] + (returns_matrix[asset] * float(weights[weights.index=='MSR'][asset]))


((1 + returns_matrix[assets+['my_portfolio']]).cumprod() -1 ).plot()
plt.show()
