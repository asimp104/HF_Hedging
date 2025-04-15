import numpy as np
from scipy import integrate


def log_psi_0(t, y_t, l_star_t, r, A_star, B0_star, B_star):
    """
    Compute psi_0_t value as defined in Proposition 2
    """
    zero = np.zeros(2)

    # Computing A_star(2, 0; t, t+1)
    A_star_val = A_star(2, zero, t, t+1)
    
    # Computing B0_star(2, 0; t, t+1)
    B0_star_val = B0_star(2, zero, t, t+1)
    
    # Computing B_star(2, 0; t, t+1)
    B_star_val = B_star(2, zero, t, t+1)
    

    dot_product = np.dot(B_star_val, l_star_t)

    log_term = -2*r + A_star_val + B0_star_val * y_t + dot_product

    return log_term
    
def log_psi_1(t, z, y_t, l_star_t, A_star, B0_star, B_star, T):
    """
    Compute psi_1_t(z) value as defined in Proposition 2
    """
    zero = np.zeros(2)

    # Computing A_star(z, 0; t, T)
    A_star_val = A_star(z, zero, t, T)
    
    # Computing B0_star(z, 0; t, T)
    B0_star_val = B0_star(z, zero, t, T)
    
    # Computing B_star(z, 0; t, T)
    B_star_val = B_star(z, zero, t, T)
    

    dot_product = np.dot(B_star_val, l_star_t)

    
    log_term = A_star_val + B0_star_val * y_t + dot_product

    return log_term


def log_psi_2(t, z, y_t, l_star_t, r, A_star, B0_star, B_star, T):
    """
    Compute psi_2_t(z) value as defined in Proposition 2
    """
    zero = np.zeros(2)

    # Computing A_star(z, 0; t+1, T)
    A_star_val_1 = A_star(z, zero, t+1, T)
    
    # Computing B0_star(z, 0; t+1, T)
    B0_star_val = B0_star(z, zero, t+1, T)
    
    # Computing B_star(z, 0; t+1, T)
    B_star_val = B_star(z, zero, t+1, T)
    
    # Computing A_star(1 + z + B0_star(z, 0; t+1, T), B_star(z, 0; t+1, T); t, t+1)
    A_star_val_2 = A_star(1 + z + B0_star_val, B_star_val, t, t+1)
    
    # Computing B0_star(1 + z + B0_star(z, 0; t+1, T), B_star(z, 0; t+1, T); t, t+1)
    B0_star_val_2 = B0_star(1 + z + B0_star_val, B_star_val, t, t+1)
    
    # Computing B_star(1 + z + B0_star(z, 0; t+1, T), B_star(z, 0; t+1, T); t, t+1)
    B_star_val_2 = B_star(1 + z + B0_star_val, B_star_val, t, t+1)
    

    dot_product = np.dot(B_star_val_2, l_star_t)

    log_term1 = -r + A_star_val_1 + A_star_val_2
    log_term2 = B0_star_val_2 * y_t
    log_term3 = dot_product
    
    # Computing psi_2_t(z)
    log_term = log_term1 + log_term2 + log_term3
    
    return log_term

def f_check_call(z, K):
    """
    Inverse Laplace transform of a European call option payoff
    
    Parameters:
    z: complex variable for the integration
    K: strike price
    
    Returns:
    Inverse Laplace transform value
    """
    return K**(1-z) / (z * (z - 1))

def f_check_put(z, K):
    """
    Inverse Laplace transform of a European put option payoff
    
    Parameters:
    z: complex variable for the integration
    K: strike price
    
    Returns:
    Inverse Laplace transform value
    """
    return K**(1-z) / (z * (z - 1)) * (-1)  # The put option payoff has opposite sign

def compute_l_t_star(s_t, q_t, delta):
    """
    Compute the scaled factor vector at time t
    
    Parameters:
    s_t: Short-run component at time t
    q_t: Long-run component at time t
    delta: Scaling factor
    
    Returns:
    l_t_star: Scaled factor vector at time t
    """
    l_t_star = np.array([s_t, q_t]) * delta
    return l_t_star

def risk_minimizing_hedge(t, T, y_t, l_star_t, r, A_star, B0_star, B_star, f_check, option_params, R):
    """
    Compute the risk-minimizing hedging position using equation (8)
    """
    # Computing psi_0_t
    log_psi_0_val = log_psi_0(t, y_t, l_star_t, r, A_star, B0_star, B_star)
        
    # Defining the integrand function for numerical integration
    def integrand_for_quad(y):
        z = complex(R, y)
        # Calculate components directly here for debugging
        log_psi_1_val = log_psi_1(t, z, y_t, l_star_t, A_star, B0_star, B_star, T)
        log_psi_2_val = log_psi_2(t, z, y_t, l_star_t, r, A_star, B0_star, B_star, T)

        psi_diff = np.exp(log_psi_2_val) - np.exp(log_psi_1_val)
        exp_term = np.exp((z-1) * y_t)
        f_val = f_check(z, **option_params)


        # Calculate the integrand value
        integrand_value = exp_term * psi_diff * f_val
        
        # Since dz = i*dy, and we have 1/(2πi) in front, we need the real part of integrand_value
        # The factor i from dz cancels with the i in the denominator
        return integrand_value.real
    
    # Integrate over smaller segments for better diagnostics
    segments = [(-100, -50), (-50, -10), (-10, -1), (-1, 1), (1, 10), (10, 50), (50, 100)]
    total_integral = 0
    
    for start, end in segments:
        segment_result, _ = integrate.quad(integrand_for_quad, start, end, limit=1000)

        total_integral += segment_result

    num = np.exp(-r * (T - t)) 

    
    # Computing the final result
    xi_t_plus_1 = np.exp(-r * (T - t)) / (2 * (np.exp(log_psi_0_val) - 1) * np.pi) * total_integral
    
    return xi_t_plus_1

def option_price(t, T, y_t, l_star_t, r, A_star, B0_star, B_star, f_check, option_params, R):
    """
    Compute the option price using equation (9)
    
    Parameters:
    t: current time point
    T: option maturity
    y_t: log of asset price at time t
    l_star_t: scaled factor vector at time t
    r: risk-free rate
    A_star, B0_star, B_star: Functions to compute the coefficients
    f_check: Inverse Laplace transform of the option payoff
    option_params: Parameters for the option payoff (e.g., strike price)
    R: Real part of the contour in the complex plane
    
    Returns:
    Option price
    """
    # Defining the complex contour for numerical integration
    # We'll use a contour that goes vertically through R + i*y for y in [-N, N]
    N = 100  # Limit for numerical integration
    
    # Defining the integrand function for the given parameters
    def price_integrand(y):
        z = complex(R, y)
        # Computing e^{zY_t}
        exp_term = np.exp(z * y_t)
        
        # Computing ψ^{(1)}_t(z)
        psi_1_val = np.exp(log_psi_1(t, z, y_t, l_star_t, A_star, B0_star, B_star, T))
        
        # Computing \check{f}(z)
        f_check_val = f_check(z, **option_params)
        
        # Computing the integrand
        result = exp_term * psi_1_val * f_check_val
        
        return result.real
    
    # Computing the integral using numerical integration
    integral_result, _ = integrate.quad(price_integrand, -N, N, limit=1000)
    
    # Computing the final result
    price = np.exp(-r * (T - t)) * integral_result / (2 * np.pi)
    
    return price