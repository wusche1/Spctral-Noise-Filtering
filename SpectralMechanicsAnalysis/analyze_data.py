import numpy as np
#import minimize
from scipy.optimize import minimize
from .models import G_Maxwell, G_Kelvin_Voigt, G_fractional_Kelvin_Voigt, PSD
from collections import Counter
import copy

def powerspectrum(trajectory, dt, k_max=None):
    # Compute the FFT of the trajectory
    trajectory_fft = np.fft.fft(trajectory)
    if k_max != None:
        k_max = min(k_max, len(trajectory) // 2)
    else:
        k_max = len(trajectory) // 2

    f_ks = np.arange(1, k_max+1) /(len(trajectory) * dt)

    frequncy_indeces = np.arange(1, k_max+1)
    
    # Only consider up to k_max frequencies
    truncated_fft = trajectory_fft[frequncy_indeces]
    
    # Compute the power spectrum
    power_spectrum =dt* np.abs(truncated_fft)**2 /len(trajectory)

        
    return f_ks, power_spectrum

def Laplace_NLL(params, x_data, y_data, function, log_weighted = False):
    return np.sum(Laplace_NLL_array(params, x_data, y_data, function, log_weighted = log_weighted))

def Laplace_NLL_array(params, x_data, y_data, function, log_weighted = False):
    y_model = function(x_data, *params)
    if log_weighted:
        weights = 1.0 / np.arange(1, len(x_data) + 1)
    Loss = y_data / y_model + np.log(y_model)
    #where the loss in NONE, set it to infinity
    Loss = np.where(np.isnan(Loss), np.inf, Loss)
    NLL = Loss
    if log_weighted:
        NLL = NLL*weights
    return NLL

def get_surprise(x_data, y_data, fitted_function, params, log_weighted = False):
    # Compute model predictions for all data points at once
    y_model = fitted_function(x_data, params)
    
    # Compute NLL for all data points at once
    Loss = y_data / y_model + np.log(y_model)

    # Compute expected NLL for all data points at once
    expected_Loss = 1 + np.log(y_model)

    if log_weighted:
        weights = 1.0 / np.arange(1, len(x_data) + 1)
        expected_Loss = expected_Loss*weights

    # Calculate surprise for all data points at once
    surprise = (Loss - expected_Loss)

    
    return surprise * np.log(2)


def fit_function(x, a, b, c):
    return a*x**b + c

def initial_guess_maxwell(x_data, y_data):
    noise = np.mean(y_data[-10:])
    A_guesses = []
    for x,y in zip(x_data[:10], y_data[:10]):
        A_guesses.append(y*x**2/2)
    A = np.mean(A_guesses)
    params = np.array([A, 1,noise])
    return params
def fit_maxwell(x_data, y_data, initial_guess = None, log_weighted = False):
    if initial_guess is None:
        initial_guess = initial_guess_maxwell(x_data, y_data)
    initial_guess = np.log(initial_guess)
    def target_funciton(x,*params):
        params = np.exp(params)
        return PSD(x,G_Maxwell,params)
    result = minimize(lambda params: Laplace_NLL(params, x_data, y_data, target_funciton, log_weighted=log_weighted), 
                  initial_guess, 
                  method='Nelder-Mead')
    fitted_params = np.exp(result.x)
    param_uncertainties = np.sqrt(np.diag(result.hess_inv))
    return fitted_params, param_uncertainties

def initial_guess_kelvin_voigt(x_data, y_data, log_weighted = False):
    noise = np.mean(y_data[-10:])
    B_over_A_square_2 = np.mean(y_data[:10])

    def localy_linear_approximation(x, noise,B_over_A_square_2,offset):
            result = np.minimum(B_over_A_square_2, np.maximum(noise, offset / x**2))
            return result
    begin = np.exp(np.log(np.min(x_data)) + (np.log(np.max(x_data)) - np.log(np.min(x_data)))/3)
    offset_initial = B_over_A_square_2 * (begin**2)
    p_0 = [noise,B_over_A_square_2, offset_initial]
    result = minimize(lambda params: Laplace_NLL(params, x_data, y_data, localy_linear_approximation, log_weighted=log_weighted), 
                  p_0, 
                  method='Nelder-Mead')
    [noise,B_over_A_square_2, offset] = result.x
    B=2/offset
    A = 1/np.sqrt(0.5*B_over_A_square_2/B)

    params = np.array([A, B, noise])
    return params

def fit_kelvin_voigt(x_data, y_data, initial_guess = None, log_weighted = False ):
    if initial_guess is None:
        initial_guess = initial_guess_kelvin_voigt(x_data, y_data)
    # initial_guess set to 0 where it is negative
    initial_guess = np.maximum(0, initial_guess)
    initial_guess = np.log(initial_guess)
    def target_funciton(x,*params):
        params = np.exp(params)
        return PSD(x,G_Kelvin_Voigt,params)
    result = minimize( lambda params: Laplace_NLL(params, x_data, y_data, target_funciton, log_weighted = log_weighted), initial_guess, method='Nelder-Mead')
    fitted_params = np.exp(result.x)
    param_uncertainties = np.sqrt(np.diag(result.hess_inv))
    return fitted_params, param_uncertainties
def initial_guess_localy_linear_fractional_kelvin_voigt(x_data, y_data):
    quaters = np.logspace(np.log10(min(x_data)), np.log10(max(x_data)), 4)
    quater_values =  [np.mean(y_data[np.argsort(np.abs(x_data - q))[:10]]) for q in quaters]

    slope_1 = np.log(quater_values[1]/quater_values[0])/np.log(quaters[1]/quaters[0])
    factor_1_0 = quater_values[0]*quaters[0]**(-slope_1)
    slope_2 = np.log(quater_values[2]/quater_values[1])/np.log(quaters[2]/quaters[1])
    factor_2_0 = quater_values[1]*quaters[1]**(-slope_2)
    noise_0 = quater_values[3]
    p_0 = [slope_1,slope_2,factor_1_0,factor_2_0,noise_0]
    return p_0
def localy_linear_approximation(x, slope_1,slope_2,factor_1,factor_2,noise):
    return np.minimum(factor_1*x**slope_1, np.maximum(noise, factor_2*x**slope_2))

def initial_guess_fractional_kelvin_voigt(x_data, y_data):
    [slope_1,slope_2,factor_1,factor_2,noise] = initial_guess_localy_linear_fractional_kelvin_voigt(x_data, y_data)
    beta = -1-slope_2
    B = np.sin(np.pi*beta/2)/factor_2
    alpha = -1-slope_1
    A = np.sin(np.pi*alpha/2)/factor_1
    if alpha < 0:
        alpha = (beta - slope_1- 1)/2
        A = np.sqrt(B*np.sin(np.pi*beta/2)/factor_1)
    if alpha < 0:
        alpha = 0
    return np.array([A,B,alpha,beta,noise])

def sigmoid(x):
    """
    Compute the sigmoid function, mapping real numbers to the interval (0, 1).
    
    Parameters:
    - x: A scalar or numpy array of any size.
    
    Returns:
    - The sigmoid function applied to every element of x.
    """
    return 1 / (1 + np.exp(-x))

def inverse_sigmoid(y):
    """
    Compute the inverse of the sigmoid function, mapping numbers from the interval (0, 1) back to real numbers.
    
    Parameters:
    - y: A scalar or numpy array of any size within the interval (0, 1).
    
    Returns:
    - The value x for which the sigmoid of x equals y.
    """
    # To avoid division by zero or log of zero, ensure y is within (0, 1)
    y = np.clip(y, 1e-10, 1 - 1e-10)
    return np.log(y / (1 - y))

def fit_fractional_kelvin_voigt(x_data, y_data, initial_guess = None, log_weighted = False):
    if initial_guess is None:
        initial_guess = initial_guess_fractional_kelvin_voigt(x_data, y_data)
    #pysical_constraints = ({'type': 'ineq', 'fun': lambda x: np.min(x)},{'type': 'ineq', 'fun': lambda x: min([1-x[2],1-x[3]])})
    A,B,alpha,beta,noise = initial_guess
    A = np.clip(A, 1e-10, np.inf)
    B = np.clip(B, 1e-10, np.inf)
    alpha = np.clip(alpha,1e-10, 1-1e-10)
    beta = np.clip(beta, 1e-10, 1-1e-10)
    noise = np.clip(noise, 1e-10, np.inf)
    initial_guess = [np.log(A),np.log(B),inverse_sigmoid(alpha),inverse_sigmoid(beta),np.log(noise)]

    def target_funciton(x,*params):
        params = [np.exp(params[0]),np.exp(params[1]),sigmoid(params[2]),sigmoid(params[3]),np.exp(params[4])]
        return PSD(x,G_fractional_Kelvin_Voigt,params)
    #result_COBYLA = minimize(Laplace_NLL, initial_guess, args=(x_data, y_data, target_funciton), method='COBYLA', constraints=pysical_constraints)
    result = minimize(lambda params: Laplace_NLL(params, x_data, y_data, target_funciton, log_weighted = log_weighted), initial_guess, method='Nelder-Mead')
    #minimize(Laplace_NLL,initial_guess, args=(x_data, y_data, target_funciton), method='Nelder-Mead')
    #if min(result.x) < 0 or min([1-result.x[2],1-result.x[3]]) < 0:
    #    result = result_COBYLA
    fitted_params = result.x
    fitted_params = [np.exp(fitted_params[0]),np.exp(fitted_params[1]),sigmoid(fitted_params[2]),sigmoid(fitted_params[3]),np.exp(fitted_params[4])]
    param_uncertainties = np.sqrt(np.diag(result.hess_inv))
    return fitted_params, param_uncertainties



def get_peak_indices(peaks):
    peak_inices = []
    for start, end in peaks:
        peak_inices += list(range(start, end+1))
    return np.array(peak_inices).astype(int)


def handle_peak_overlap(peaks):
    peak_boundaries = []
    for start, end in peaks:
        peak_boundaries.append(start)
        peak_boundaries.append(end)
    boundary_counter = Counter(peak_boundaries)
    peak_boundaries =  [boundary for boundary, count in boundary_counter.items() if count % 2 != 0]
    peak_boundaries = sorted(peak_boundaries)

    new_peaks = []
    for i in range(len(peak_boundaries)//2):
        new_peaks.append((peak_boundaries[2*i], peak_boundaries[2*i+1]))
    return new_peaks

def find_maximum_difference(arr: np.ndarray):
    if len(arr) < 2:
        return 0  # No possible pairs if array has less than 2 elements

    min_element_index = 0
    max_diff = 0
    max_diff_pair = (0, 0)

    for i in range(1, len(arr)):
        if arr[i] < arr[min_element_index]:
            min_element_index = i
        elif arr[i] - arr[min_element_index] > max_diff:
            max_diff = arr[i] - arr[min_element_index]
            max_diff_pair = (min_element_index, i)

    return max_diff_pair, max_diff

def find_max_evidence_peak(peaks,surprise, max_peak_percentage):
    #print(max_peak_percentage)
    assert 0 <= max_peak_percentage <= 1, "max_peak_percentage must be between 0 and 1"
    if max_peak_percentage == 1:
        prior = 0
    else:
        prior = max(0,sorted(surprise)[int(len(surprise)*max_peak_percentage)])
    peak_ineces = get_peak_indices(peaks)
    sup = copy.deepcopy(surprise)-prior
    sup[peak_ineces] = -sup[peak_ineces]
    cumsum = np.cumsum(sup)
    peak, evidence = find_maximum_difference(cumsum)
    return peak, evidence+prior
def find_all_peaks(surprise, max_peak_percentage, typical_peak_number):
    peaks = []
    while True:
        peak, evidence = find_max_evidence_peak(peaks,surprise, max_peak_percentage)
        if evidence -np.log2(len(surprise))-np.log2(typical_peak_number) > len(peaks)/typical_peak_number:
            peaks.append(peak)
            peaks = handle_peak_overlap(peaks)
        else:
            break
    return peaks