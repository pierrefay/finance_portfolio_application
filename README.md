# finance_portfolio_application
exercices d'application utilisé pour un TP.

Ennoncé exercice 1:
 - à partir de Yahoo Finance téléchargez les données OHLCV
 - calculer les returns
 - calculer le return du SP500 qu'on prendra comme indice de marché
 - calculer le BETA d'un asset
 - calculer le return attendu a partir du CAPM (MEDAF)
 - afficher la SML
 - calculer les ratio de sharpe, traynor et jensen

Ennoncé exercice 2:
- voir ensemble comment on trace la frontiere efficiente
- voir ensemble comment on calcule les portefeuilles suivants : MSR (max sharpe ratio), GMV (global minimum variance)


pour démarrer, il faut installer l'environnement, voici les commandes à taper dans votre console:
pip install virtualenv
virtualenv test
pip install -r path/to/requirements.txt

pour lancer l'exo1:
python factors.py

pour l'exo2:
python porftolio.py




Inspiré par https://www.coursera.org/learn/advanced-portfolio-construction-python/
