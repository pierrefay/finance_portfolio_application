import math

def black_scholes_call(S, K, r, T, sigma):
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    N1 = 0.5*(1 + math.erf(d1/math.sqrt(2)))
    N2 = 0.5*(1 + math.erf(d2/math.sqrt(2)))
    call_price = S*N1 - K*math.exp(-r*T)*N2
    return call_price

def black_scholes_put(S, K, r, T, sigma):
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    N1 = 0.5*(1 + math.erf(-d1/math.sqrt(2)))
    N2 = 0.5*(1 + math.erf(-d2/math.sqrt(2)))
    put_price = K*math.exp(-r*T)*N2 - S*N1
    return put_price

def black_scholes_strike(S, price, r, T, sigma, is_call=True):
	if is_call:
	    return black_scholes_call(S, K, r, T, sigma) - price
	else:
	    return black_scholes_put(S, K, r, T, sigma) - price

    

S=69
K=70
r=0.05
T=0.5
sigma=0.35

results_put = black_scholes_put(S, K, r, T, sigma)
print("put : " +str(results_put))

results_call = black_scholes_call(S, K, r, T, sigma)
print("call : " +str(results_call))
