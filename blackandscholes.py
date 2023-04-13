import math
from scipy.stats import norm
import numpy as np

def black_scholes_call(S, K, r, T, sigma):
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    print("call d1 :"+str(round(d1,4)))
    d2 = d1 - sigma*math.sqrt(T)
    print("call d2 :"+str(round(d2,4)))
    N1 = 0.5*(1 + math.erf(d1/math.sqrt(2)))
    print("call N1 :"+str(round(N1,4)))
    N2 = 0.5*(1 + math.erf(d2/math.sqrt(2)))
    print("call N2 :"+str(round(N2,4)))
    call_price = S*N1 - K*math.exp(-r*T)*N2
    return call_price

def black_scholes_put(S, K, r, T, sigma):
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    print("put d1 :"+str(round(d1,4)))
    d2 = d1 - sigma*math.sqrt(T)
    print("put d2 :"+str(round(d2,4)))
    N1 = 0.5*(1 + math.erf(-d1/math.sqrt(2)))
    print("put N1 :"+str(round(N1,4)))
    N2 = 0.5*(1 + math.erf(-d2/math.sqrt(2)))
    print("put N2 :"+str(round(N2,4)))
    put_price = K*math.exp(-r*T)*N2 - S*N1
    return put_price

def black_scholes_strike(S, price, r, T, sigma, is_call=True):
	if is_call:
	    return black_scholes_call(S, K, r, T, sigma) - price
	else:
	    return black_scholes_put(S, K, r, T, sigma) - price

def d1(S, K, r, sigma, T):
    return (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))

def d2(S, K, r, sigma, T):
    return d1(S, K, r, sigma, T) - sigma * np.sqrt(T)

def call_delta(S, K, r, sigma, T):
    return norm.cdf(d1(S, K, r, sigma, T))

def put_delta(S, K, r, sigma, T):
    return -norm.cdf(-d1(S, K, r, sigma, T))

def call_gamma(S, K, r, sigma, T):
    return norm.pdf(d1(S, K, r, sigma, T)) / (S * sigma * np.sqrt(T))

def put_gamma(S, K, r, sigma, T):
    return call_gamma(S, K, r, sigma, T)

def call_theta(S, K, r, sigma, T):
    d1_value = d1(S, K, r, sigma, T)
    d2_value = d2(S, K, r, sigma, T)
    return -(S * norm.pdf(d1_value) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2_value)

def put_theta(S, K, r, sigma, T):
    d1_value = d1(S, K, r, sigma, T)
    d2_value = d2(S, K, r, sigma, T)
    return -(S * norm.pdf(d1_value) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2_value)

def vega(S, K, r, sigma, T):
    return S * norm.pdf(d1(S, K, r, sigma, T)) * np.sqrt(T)

def call_rho(S, K, r, sigma, T):
    d2_value = d2(S, K, r, sigma, T)
    return K * T * np.exp(-r * T) * norm.cdf(d2_value)

def put_rho(S, K, r, sigma, T):
    d2_value = d2(S, K, r, sigma, T)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2_value)
    
S=12
K=10
r=0.04
T=0.3
sigma=0.1

results_put = black_scholes_put(S, K, r, T, sigma)
print("put : " +str(round(results_put,4)))
print("put delta:" +str(round(put_delta(S, K, r, sigma, T),2)))
print("put gamma:" +str(round(put_gamma(S, K, r, sigma, T),2)))
print("put theta:" +str(round(put_theta(S, K, r, sigma, T),2)))
print("put rho:" +str(round(put_rho(S, K, r, sigma, T),2)))

results_call = black_scholes_call(S, K, r, T, sigma)
print("call : " +str(round(results_call,4)))
print("call delta:" +str(round(call_delta(S, K, r, sigma, T),2)))
print("call gamma:" +str(round(call_gamma(S, K, r, sigma, T),2)))
print("call theta:" +str(round(call_theta(S, K, r, sigma, T),2)))
print("call rho:" +str(round(call_rho(S, K, r, sigma, T),2)))

