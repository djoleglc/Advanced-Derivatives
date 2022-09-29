
def black_scholes_call(t, T, s, k, r, q, sigma):
    d1 = ( np.log(s/k) + (r - q + 0.5*sigma**2)*(T-t) )/(sigma * np.sqrt(T-t))
    d2 = d1 - sigma * np.sqrt(T-t)
    price = s * np.exp(-q *(T-t))*scipy.stats.norm.cdf(d1) - k*np.exp(-r *(T-t))*scipy.stats.norm.cdf(d2)
    return price
  
  def call(s, k):
    return (s-k > 0)*(s-k)
  
  
def simulation_option_final(function, t, T, s, r, q, sigma, n):
    tau = T-t
    x = np.random.normal(0,1,n)
    s_T = s*np.exp((r - q -0.5*sigma**2)*tau + sigma*x*np.sqrt(tau))
    fun = function(s_T)
    
    return np.exp(-r*tau)*fun.mean()
  
  
  def gbm(x, t, T, s, k, r, q, sigma):
    tau = T-t
    s_T = s*np.exp((r - q -0.5*sigma**2)*tau + sigma*x*np.sqrt(tau)) - k
    return np.exp( -r*(T-t)) * f( s_T) * np.exp(-0.5*x**2)*(1/np.sqrt(2*math.pi))
