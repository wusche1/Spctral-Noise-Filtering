import matplotlib.pyplot as plt
from analyze_data import *
import numpy as np

class Data:
    def __init__(self, t, x, name=None, typical_peak_number = 0.05, max_peak_percentage = .95,prior_maxwell=1/3, prior_kelvin_voigt=1/3, prior_fractional_kelvin_voigt=1/3):
        self.t = t
        self.x = x
        self.name = name
        self.PSD = None
        self.frequencies = None

        self.initial_guess_maxwell = None
        self.fit_maxwell = None
        self.NLL_maxwell = None
        self.initial_guess_kelvin_voigt = None
        self.fit_kelvin_voigt = None
        self.NLL_kelvin_voigt = None
        self.initial_guess_fractional_kelvin_voigt = None
        self.fit_fractional_kelvin_voigt = None
        self.NLL_fractional_kelvin_voigt = None

        self.prior_maxwell = prior_maxwell
        self.prior_kelvin_voigt = prior_kelvin_voigt
        self.prior_fractional_kelvin_voigt = prior_fractional_kelvin_voigt

        self.posterior_maxwell = None
        self.posterior_kelvin_voigt = None
        self.posterior_fractional_kelvin_voigt = None


        self.typical_peak_number = typical_peak_number
        self.max_peak_percentage = max_peak_percentage 

        self.fit_function = None
        self.fit_params = None
        self.peaks = []
        self.suprise = None
        



    def create_PSD(self):
        # Implement the logic to create the PSD
        if self.PSD is not None:
            return
        dt = self.t[1] - self.t[0]
        fq, ps = powerspectrum(self.x, dt)
        self.PSD = ps
        self.frequencies = fq

    def create_initial_guess_maxwell(self):
        self.create_PSD()
        # Implement the logic to create the initial guess for the Maxwell model
        self.initial_guess_maxwell = initial_guess_maxwell(self.frequencies, self.PSD)

    def create_fit_maxwell(self):
        # Implement the logic to create the fit for the Maxwell model
        self.create_PSD()  # Create the PSD first
        if self.initial_guess_maxwell is None:
            self.create_initial_guess_maxwell()
        self.fit_maxwell = fit_maxwell(self.frequencies, self.PSD, self.initial_guess_maxwell)

    def create_NLL_maxwell(self):
        # Implement the logic to create the NLL for the Maxwell model
        if self.fit_maxwell is None:
            self.create_fit_maxwell()
        target_function = lambda x, *params: PSD(x, G_Maxwell, params)
        self.NLL_maxwell = Laplace_NLL(self.fit_maxwell.x, self.frequencies, self.PSD, target_function)

    def create_initial_guess_kelvin_voigt(self):
        # Implement the logic to create the initial guess for the Kelvin-Voigt model
        self.initial_guess_kelvin_voigt = initial_guess_kelvin_voigt(self.frequencies, self.PSD)

    def create_fit_kelvin_voigt(self):
        # Implement the logic to create the fit for the Kelvin-Voigt model
        self.create_PSD()  # Create the PSD first
        if self.initial_guess_kelvin_voigt is None:
            self.create_initial_guess_kelvin_voigt()
        self.fit_kelvin_voigt = fit_kelvin_voigt(self.frequencies, self.PSD, self.initial_guess_kelvin_voigt)

    def create_NLL_kelvin_voigt(self):
        # Implement the logic to create the NLL for the Kelvin-Voigt model
        if self.fit_kelvin_voigt is None:
            self.create_fit_kelvin_voigt()
        target_function = lambda x, *params: PSD(x, G_Kelvin_Voigt, params)
        self.NLL_kelvin_voigt = Laplace_NLL(self.fit_kelvin_voigt.x, self.frequencies, self.PSD, target_function)

    def create_initial_guess_fractional_kelvin_voigt(self):
        # Implement the logic to create the initial guess for the Fractional Kelvin-Voigt model
        self.initial_guess_fractional_kelvin_voigt = initial_guess_fractional_kelvin_voigt(self.frequencies, self.PSD)

    def create_fit_fractional_kelvin_voigt(self):
        # Implement the logic to create the fit for the Fractional Kelvin-Voigt model
        self.create_PSD()  # Create the PSD first
        if self.initial_guess_fractional_kelvin_voigt is None:
            self.create_initial_guess_fractional_kelvin_voigt()
        self.fit_fractional_kelvin_voigt = fit_fractional_kelvin_voigt(self.frequencies, self.PSD,
                                                                       self.initial_guess_fractional_kelvin_voigt)

    def create_NLL_fractional_kelvin_voigt(self):
        # Implement the logic to create the NLL for the Fractional Kelvin-Voigt model
        if self.fit_fractional_kelvin_voigt is None:
            self.create_fit_fractional_kelvin_voigt()
        target_function = lambda x, *params: PSD(x, G_fractional_Kelvin_Voigt, params)
        self.NLL_fractional_kelvin_voigt = Laplace_NLL(self.fit_fractional_kelvin_voigt.x, self.frequencies, self.PSD,
                                                       target_function)

    def bayesian_update(self):
        if self.NLL_maxwell is None:
            self.create_NLL_maxwell()
        if self.NLL_kelvin_voigt is None:
            self.create_NLL_kelvin_voigt()
        if self.NLL_fractional_kelvin_voigt is None:
            self.create_NLL_fractional_kelvin_voigt()

        # Implement the logic to update the prior probabilities
        # Use the Bayesian Information Criterion to calculate the posterior probability

        BIC_maxwell = 2 * np.log(len(self.PSD)) + 2 * self.NLL_maxwell
        BIC_kelvin_voigt = 3 * np.log(len(self.PSD)) + 2 * self.NLL_kelvin_voigt
        BIC_fractional_kelvin_voigt = 5 * np.log(len(self.PSD)) + 2 * self.NLL_fractional_kelvin_voigt

        #subtract the minimum BIC from all BICs
        min_BIC = min(BIC_maxwell, BIC_kelvin_voigt, BIC_fractional_kelvin_voigt)
        BIC_maxwell -= min_BIC
        BIC_kelvin_voigt -= min_BIC
        BIC_fractional_kelvin_voigt -= min_BIC

        # Calculate the posterior probability
        posterior_maxwell = np.exp(-BIC_maxwell / 2) * self.prior_maxwell
        posterior_kelvin_voigt = np.exp(-BIC_kelvin_voigt / 2) * self.prior_kelvin_voigt
        posterior_fractional_kelvin_voigt = np.exp(-BIC_fractional_kelvin_voigt / 2) * self.prior_fractional_kelvin_voigt

        # Normalize the posterior probability
        self.posterior_maxwell = posterior_maxwell / (posterior_maxwell + posterior_kelvin_voigt + posterior_fractional_kelvin_voigt)
        self.posterior_kelvin_voigt = posterior_kelvin_voigt / (posterior_maxwell + posterior_kelvin_voigt + posterior_fractional_kelvin_voigt)
        self.posterior_fractional_kelvin_voigt = posterior_fractional_kelvin_voigt / (posterior_maxwell + posterior_kelvin_voigt + posterior_fractional_kelvin_voigt)

        if self.posterior_maxwell > self.posterior_kelvin_voigt and self.posterior_maxwell > self.posterior_fractional_kelvin_voigt:
            self.fit_function = lambda x, params: PSD(x, G_Maxwell, params)
            self.fit_params = self.fit_maxwell.x
        elif self.posterior_kelvin_voigt > self.posterior_maxwell and self.posterior_kelvin_voigt > self.posterior_fractional_kelvin_voigt:
            self.fit_function = lambda x, params: PSD(x, G_Kelvin_Voigt, params)
            self.fit_params = self.fit_kelvin_voigt.x
        else:
            self.fit_function = lambda x, params: PSD(x, G_fractional_Kelvin_Voigt, params)
            self.fit_params = self.fit_fractional_kelvin_voigt.x
        
        self.suprise = get_surprise(self.frequencies, self.PSD, self.fit_function, self.fit_params)
    
    def refit(self):
        #filter peaks out of data
        idx_to_delete = np.array([])
        for peak in self.peaks:
            idx_to_delete = np.append(idx_to_delete, np.arange(peak[0], peak[1]+1))
        idx_to_delete = idx_to_delete.astype(int)
        filtered_idx = np.arange(len(self.frequencies))
        filtered_idx = np.delete(filtered_idx, idx_to_delete)
        filtered_freq = self.frequencies[filtered_idx]
        filtered_PSD = self.PSD[filtered_idx]

        if self.posterior_fractional_kelvin_voigt > self.posterior_maxwell and self.posterior_fractional_kelvin_voigt > self.posterior_kelvin_voigt:
            self.fit_params = fit_fractional_kelvin_voigt(filtered_freq, filtered_PSD, self.fit_params).x
        elif self.posterior_maxwell > self.posterior_fractional_kelvin_voigt and self.posterior_maxwell > self.posterior_kelvin_voigt:
            self.fit_params = fit_maxwell(filtered_freq, filtered_PSD, self.fit_params).x
        else:
            self.fit_params = fit_kelvin_voigt(filtered_freq, filtered_PSD, self.fit_params).x
        self.suprise = get_surprise(self.frequencies, self.PSD, self.fit_function, self.fit_params)
    
    def find_peaks(self, max_iter=10, report = False):
        self.peaks = find_all_peaks(self.suprise, self.max_peak_percentage,self.typical_peak_number)
        iter = 0
        while iter < max_iter:
            iter += 1
            if report:
                print("Iteration: ", iter)
            old_peaks = self.peaks.copy()
            self.refit()
            new_peaks = find_all_peaks(self.suprise,self.max_peak_percentage, self.typical_peak_number)
            if old_peaks == new_peaks:
                return
        

    def plot_psd(self, ax=None):
        def log_spaced_frequencies(frequencies, num_points=1000):
            min_freq = np.min(frequencies)
            max_freq = np.max(frequencies)
            log_min_freq = np.log10(min_freq)
            log_max_freq = np.log10(max_freq)
            log_spaced_freqs = np.logspace(log_min_freq, log_max_freq, num=num_points)
            return log_spaced_freqs
        # Implement the logic to plot the PSD, initial guesses, and fits
        self.create_PSD()  # Create the PSD first

        if ax is None:
            ax = plt.gca()

        # Plot the PSD
        ax.scatter(self.frequencies, self.PSD, label='PSD', s=0.5)

        # Generate logarithmically spaced frequencies
        log_spaced_freqs = log_spaced_frequencies(self.frequencies)

        # Plot the initial guesses and fits for the Maxwell model
        if self.initial_guess_maxwell is not None and self.fit_maxwell is not None:
            psd_initial_guess_maxwell = PSD(log_spaced_freqs, G_Maxwell, self.initial_guess_maxwell)
            ax.plot(log_spaced_freqs, psd_initial_guess_maxwell, label='Initial Guess (Maxwell)', linestyle='--')
            psd_fit_maxwell = PSD(log_spaced_freqs, G_Maxwell, self.fit_maxwell.x)
            ax.plot(log_spaced_freqs, psd_fit_maxwell, label='Fit (Maxwell)')

        # Plot the initial guesses and fits for the Kelvin-Voigt model
        if self.initial_guess_kelvin_voigt is not None and self.fit_kelvin_voigt is not None:
            psd_initial_guess_kelvin_voigt = PSD(log_spaced_freqs, G_Kelvin_Voigt, self.initial_guess_kelvin_voigt)
            ax.plot(log_spaced_freqs, psd_initial_guess_kelvin_voigt, label='Initial Guess (Kelvin-Voigt)',
                     linestyle='--')
            psd_fit_kelvin_voigt = PSD(log_spaced_freqs, G_Kelvin_Voigt, self.fit_kelvin_voigt.x)
            ax.plot(log_spaced_freqs, psd_fit_kelvin_voigt, label='Fit (Kelvin-Voigt)')

        # Plot the initial guesses and fits for the Fractional Kelvin-Voigt model
        if self.initial_guess_fractional_kelvin_voigt is not None and self.fit_fractional_kelvin_voigt is not None:
            psd_initial_guess_fractional_kelvin_voigt = PSD(log_spaced_freqs, G_fractional_Kelvin_Voigt,
                                                            self.initial_guess_fractional_kelvin_voigt)
            ax.plot(log_spaced_freqs, psd_initial_guess_fractional_kelvin_voigt,
                     label='Initial Guess (Fractional Kelvin-Voigt)', linestyle='--')
            psd_fit_fractional_kelvin_voigt = PSD(log_spaced_freqs, G_fractional_Kelvin_Voigt,
                                                  self.fit_fractional_kelvin_voigt.x)
            ax.plot(log_spaced_freqs, psd_fit_fractional_kelvin_voigt, label='Fit (Fractional Kelvin-Voigt)')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectrum')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()


    def plot_x(self, ax=None):
        # Implement the logic to plot the time series data
        if ax is None:
            ax = plt.gca()

        ax.plot(self.t, self.x)
        class Data:
            def __init__(self, t, x, name=None, prior_maxwell=1/3, prior_kelvin_voigt=1/3, prior_fractional_kelvin_voigt=1/3):
                self.t = t
                self.x = x
                self.name = name
                self.PSD = None
                self.frequencies = None

                self.initial_guess_maxwell = None
                self.fit_maxwell = None
                self.NLL_maxwell = None
                self.initial_guess_kelvin_voigt = None
                self.fit_kelvin_voigt = None
                self.NLL_kelvin_voigt = None
                self.initial_guess_fractional_kelvin_voigt = None
                self.fit_fractional_kelvin_voigt = None
                self.NLL_fractional_kelvin_voigt = None

                self.prior_maxwell = prior_maxwell
                self.prior_kelvin_voigt = prior_kelvin_voigt
                self.prior_fractional_kelvin_voigt = prior_fractional_kelvin_voigt

                self.posterior_maxwell = None
                self.posterior_kelvin_voigt = None
                self.posterior_fractional_kelvin_voigt = None
