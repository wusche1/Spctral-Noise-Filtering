import numpy as np
#First we define the complex sheer moduli:

def G_fractional_Kelvin_Voigt(omega, A, B, alpha, beta):
    return A * (1j * omega) ** alpha + B * (1j * omega) ** beta
def G_Kelvin_Voigt(omega, A, B):
    return A  + B* (1j * omega)
def G_Maxwell(omega, A, B):
    return 1/(A /((1j * omega)) + B)
#predict the Power Spectrum from the complex sheer moduli

def PSD(omega, G, args, kbT=1):
    G_args, noise = args[:-1], args[-1]
    # Calculate PSD but use np.where to handle the case when omega is 0
    psd = np.where(omega == 0, 0, -2 * kbT / omega * np.imag(1/G(omega, *G_args)) + noise)
    return psd