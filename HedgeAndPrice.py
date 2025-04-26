import numpy as np
from mpmath import *
import matplotlib.pyplot as plt



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
    Compute the risk-minimizing hedge ratio using Proposition 2.
    
    Parameters:
    -----------
    t : int
        Current time
    T : int
        Maturity time
    y_t : float
        Current log-price
    l_star_t : array
        Current factor values
    r : float
        Risk-free rate
    A_star, B0_star, B_star : functions
        Coefficient functions for the MGF
    f_check : function
        Laplace transform of option payoff function
    option_params : dict
        Option parameters (e.g., strike price)
    R : float
        Contour parameter (typically 1.5 for calls, -0.5 for puts)
    
    Returns:
    --------
    xi_t : float
        The risk-minimizing hedge ratio (number of shares to hold)
    """

    
    # Define the integrand function using your helper functions
    def integrand(y):
        y_np = float(y)
        z = complex(R, y_np)
        
        # Compute psi_1 and psi_2 using your helper functions
        log_psi_1_val = log_psi_1(t, z, y_t, l_star_t, A_star, B0_star, B_star, T)
        log_psi_2_val = log_psi_2(t, z, y_t, l_star_t, r, A_star, B0_star, B_star, T)
        
        # Convert to values
        psi_1_val = np.exp(log_psi_1_val)
        psi_2_val = np.exp(log_psi_2_val)
        
        # Compute difference
        psi_diff = psi_2_val - psi_1_val
        
        # Exponential term
        exp_term = np.exp((z-1) * y_t)
        
        # Payoff transform
        f_val = f_check(z, **option_params)
        
        res = exp_term * psi_diff * f_val * 1j
        res_mp = mpc(res.real, res.imag)

        # Full integrand
        return res_mp

    # Perform numerical integration
    integral_result = quad(integrand, [-inf, inf], method="tanh-sinh")
    
    # Compute psi_0 (denominator term) using your helper function
    log_psi_0_val = log_psi_0(t, y_t, l_star_t, r, A_star, B0_star, B_star)
    psi_0 = np.exp(log_psi_0_val)

    # Compute final hedge ratio
    xi_t = np.exp(-r * (T-t)) * integral_result / (2 * np.pi * psi_0 * 1j)
    
    return xi_t


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

    
    # Defining the integrand function for the given parameters
    def integrand(y):
        y_np = float(y)
        z = complex(R, y_np)
        # Computing e^{zY_t}
        exp_term = np.exp(z * y_t)
        
        # Computing ψ^{(1)}_t(z)
        psi_1_val = np.exp(log_psi_1(t, z, y_t, l_star_t, A_star, B0_star, B_star, T))
        
        # Computing \check{f}(z)
        f_check_val = f_check(z, **option_params)
        
        # Computing the integrand
        result = exp_term * psi_1_val * f_check_val * 1j
        result_mp = mpc(result.real, result.imag)
        
        return result_mp
    
    # Computing the integral using numerical integration
    integral_result = quad(integrand, [-inf, inf], method="tanh-sinh")
    
    # Computing the final result
    price = np.exp(-r * (T - t)) * integral_result / (2 * np.pi * 1j)
    
    return price


def plot_func(t, T, y_t, l_star_t, r, A_star, B0_star, B_star, f_check, option_params, range):
    # y values
    y_vals = linspace(-range, range, 1000)
    R = 2.0  # adjust as needed

    # Storage
    psi1_real, psi1_imag = [], []
    psi2_real, psi2_imag = [], []
    exp_real, exp_imag = [], []
    diff_real, diff_imag = [], []
    fval_real, fval_imag = [], []

    int_real, int_imag = [], []

    res_real, res_imag = [], []

    for y in y_vals:
        z = complex(R, y)
        
        # Your functions
        log_psi_1_val = log_psi_1(t, z, y_t, l_star_t, A_star, B0_star, B_star, T)
        log_psi_2_val = log_psi_2(t, z, y_t, l_star_t, r, A_star, B0_star, B_star, T)
        
        psi_1 = exp(log_psi_1_val)
        psi_2 = exp(log_psi_2_val)
        exp_term = exp((z - 1) * y_t)
        psi_diff = psi_2 - psi_1
        f_val = f_check(z, **option_params)

        integrand = exp_term * psi_diff * f_val* 1j

        res = exp(z*y_t) * psi_1 * f_val
        res_real.append(res.real)
        res_imag.append(res.imag)

        
        # Store parts
        psi1_real.append(psi_1.real)
        psi1_imag.append(psi_1.imag)
        psi2_real.append(psi_2.real)
        psi2_imag.append(psi_2.imag)
        exp_real.append(exp_term.real)
        exp_imag.append(exp_term.imag)
        diff_real.append(psi_diff.real)
        diff_imag.append(psi_diff.imag)
        fval_real.append(f_val.real)
        fval_imag.append(f_val.imag)
        int_real.append(integrand.real)
        int_imag.append(integrand.imag)

    # Plot function
    def plot_complex_component(y_vals, real_vals, imag_vals, title):
        plt.figure(figsize=(10, 4))
        plt.plot(y_vals, real_vals, label='Real part')
        plt.plot(y_vals, imag_vals, label='Imag part', linestyle='--')
        plt.title(title)
        plt.xlabel('y')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Generate plots
    plot_complex_component(y_vals, psi1_real, psi1_imag, 'ψ₁(z)')
    plot_complex_component(y_vals, psi2_real, psi2_imag, 'ψ₂(z)')
    plot_complex_component(y_vals, exp_real, exp_imag, 'exp((z-1)·yₜ)')
    plot_complex_component(y_vals, diff_real, diff_imag, 'ψ₂(z) - ψ₁(z)')
    plot_complex_component(y_vals, fval_real, fval_imag, 'f̂(z)')
    plot_complex_component(y_vals, int_real, int_imag, 'Integrand')
    plot_complex_component(y_vals, res_real, res_imag, 'price integrand')

