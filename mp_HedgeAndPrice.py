from mpmath import mp, exp, pi, quad, quadosc, inf, mpc, matrix
import numpy as np
from scipy.stats import norm

mp.dps = 30  # Set desired precision

def log_psi_0(t, y_t, l_star_t, r, A_star, B0_star, B_star):
    zero_vec = matrix([0, 0])
    two = mpc(2, 0)
    A_star_val = A_star(two, zero_vec, t, t+1)
    B0_star_val = B0_star(two, zero_vec, t, t+1)
    B_star_val = B_star(two, zero_vec, t, t+1)
    dot_product = sum(B_star_val[i] * l_star_t[i] for i in range(2))
    log_term = -2*r + A_star_val + B0_star_val * y_t + dot_product
    return log_term

def log_psi_1(t, z, y_t, l_star_t, A_star, B0_star, B_star, T):
    zero_vec = matrix([0, 0])
    A_star_val = A_star(z, zero_vec, t, T)
    B0_star_val = B0_star(z, zero_vec, t, T)
    B_star_val = B_star(z, zero_vec, t, T)
    dot_product = sum(B_star_val[i] * l_star_t[i] for i in range(2))
    return A_star_val + B0_star_val * y_t + dot_product

def log_psi_2(t, z, y_t, l_star_t, r, A_star, B0_star, B_star, T):
    zero_vec = matrix([0, 0])
    A_star_val_1 = A_star(z, zero_vec, t+1, T)
    B0_star_val = B0_star(z, zero_vec, t+1, T)
    B_star_val = B_star(z, zero_vec, t+1, T)
    shift_z = 1 + z + B0_star_val
    A_star_val_2 = A_star(shift_z, B_star_val, t, t+1)
    B0_star_val_2 = B0_star(shift_z, B_star_val, t, t+1)
    B_star_val_2 = B_star(shift_z, B_star_val, t, t+1)
    dot_product = sum(B_star_val_2[i] * l_star_t[i] for i in range(2))
    return -r + A_star_val_1 + A_star_val_2 + B0_star_val_2 * y_t + dot_product

def f_check_call(z, K):
    return K**(1 - z) / (z * (z - 1))

def f_check_put(z, K):
    return -K**(1 - z) / (z * (z - 1))

def compute_l_t_star(s_t, q_t, delta):
    return matrix([s_t * delta, q_t * delta])

def risk_minimizing_hedge(t, T, y_t, l_star_t, r, A_star, B0_star, B_star, f_check, option_params, R):
    def integrand(y):
        z = mpc(R, y)
        log_psi1 = log_psi_1(t, z, y_t, l_star_t, A_star, B0_star, B_star, T)
        log_psi2 = log_psi_2(t, z, y_t, l_star_t, r, A_star, B0_star, B_star, T)
        psi1 = exp(log_psi1)
        psi2 = exp(log_psi2)
        psi_diff = psi2 - psi1
        f_val = f_check(z, **option_params)
        return exp((z - 1) * y_t) * psi_diff * f_val * 1j
    
    result = quadosc(integrand, [-inf, inf], omega=y_t)
    log_psi0 = log_psi_0(t, y_t, l_star_t, r, A_star, B0_star, B_star)
    psi0 = exp(log_psi0) - 1
    return exp(-r * (T - t)) * result / (2 * pi * psi0)

def option_price(t, T, y_t, l_star_t, r, A_star, B0_star, B_star, f_check, option_params, R):
    def integrand(y):
        z = mpc(R, y)
        return (exp(z * y_t)
                * exp(log_psi_1(t, z, y_t, l_star_t, A_star, B0_star, B_star, T))
                * f_check(z, **option_params)).real

    integral_result = quadosc(integrand, [inf, inf], omega=y_t)
    return exp(-r * (T - t)) * integral_result / (2 * pi)

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the price of European options using the Black-Scholes formula
    
    Parameters:
    S: Current stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate (annual)
    sigma: Volatility (annual)
    option_type: 'call' or 'put'
    
    Returns:
    Option price
    """
    # Calculate d1 and d2
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate option price
    if option_type.lower() == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return option_price
