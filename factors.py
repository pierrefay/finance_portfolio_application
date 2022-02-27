# Etape 1- recupération des données de Yahoo Finance et création de la matrice des returns
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tools

#taux sans risque
rf = 0

#liste des symbole qu'on veut recupérer
symbols = ['MSFT','AAPL','AMZN','^GSPC','PM','KO','AAL','AXP','SBUX','JNJ']
returns_matrix = pd.DataFrame()
price_matrix = pd.DataFrame()
symbolsdf = pd.DataFrame()
#on boucle sur tous ce qu'on veut récupérer pour mettre dans notre portefeuille
for symbol in symbols:
    #on recupere l'historique
    ticker = yf.Ticker(symbol)
    symboldf = ticker.history(interval='1d', start='2015-01-01')
    # symboldf = ticker.history(interval='1d', start='2020-01-01', end='2021-04-01')
    symboldf['symbol'] = symbol
    #on calcule le returns
    # on supprime la premiere ligne qui est un NaN
    # on rename la colonne "returns" en AAPL par exemple pour la stocker directement dans la matrice avec le bon nom
    symboldf['returns'] = symboldf['Close'].pct_change()
    symboldf.dropna(inplace=True)
    symbolsdf = pd.concat([symbolsdf, symboldf],axis=0)

#calculer la var des returns du marché
returns_sp500 = symbolsdf[symbolsdf.symbol=='^GSPC']['returns'] - rf
var_sp500 = returns_sp500.var()

#calculer la covariance (stock, marché) et calcul du beta
covs = symbolsdf.groupby('symbol').apply(lambda x: np.cov(x['returns'].values, returns_sp500)[0][1])
returns = symbolsdf.groupby('symbol').apply(lambda x: tools.total_returns(x['returns']))

result = pd.DataFrame({"symbol":covs.index.values, "cov_w_sp500":covs.values, 'var_sp500':var_sp500, 'returns':returns})
result['beta'] = result['cov_w_sp500']/result['var_sp500']

#calcul du required return
sp500returns = float(result[result.symbol=='^GSPC']['returns'])
result['required_returns'] = rf + result['beta']*(sp500returns - rf)


#calcul sharpe / traynor / jensen
result['sharpe'] = (result['returns'] - rf) / result['returns'].std()
result['traynor'] = (result['returns'] - rf) / result['beta']
result['jensen'] = result['returns'] - (rf + result['beta']*(sp500returns - rf))

#on affiche le tableau
print(result[['returns','beta','required_returns','sharpe','traynor','jensen']])

plt.plot(result['beta'],result['required_returns'], 'r--')
plt.scatter(result['beta'],result['returns'])

plt.show()



