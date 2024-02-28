import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def mean_back_realxation(dt, trajectory, Tao, timespan, multiplier=1, d_sigma_estimation_samples = 1000, return_n_points_added=False, progress_bar = True):
    if isinstance(trajectory, list):
        trajectory = np.array(trajectory, dtype=np.float64)  # Ensure high precision
    forward_sampling = int(timespan / dt)
    backward_sampling = int(Tao / dt)

    d_array = trajectory[backward_sampling:] - trajectory[:-backward_sampling]

    # Estimate the standard deviation of the d array
    d_sample = np.random.choice(d_array, size=min(d_sigma_estimation_samples, len(d_array)), replace=False)
    d_sigma = np.std(d_sample)
    d_cutoff = d_sigma * multiplier

    valid_points = np.abs(d_array) > d_cutoff
    valid_indices = np.where(valid_points[:-forward_sampling])[0]

    # Initialize an accumulator for the samples with high precision
    sample_accumulator = np.zeros(forward_sampling, dtype=np.float64)

    n_points_added = 0

    # Iterate over the valid indices and accumulate the samples. make a progress bar if progress_bar is True
    for idx in tqdm(valid_indices, disable=not progress_bar, total=len(valid_indices)):
        sample = trajectory[idx + backward_sampling : idx + backward_sampling + forward_sampling] - trajectory[idx + backward_sampling]
        d_mbr = d_array[idx]
        sample_to_add = (-sample / d_mbr).astype(np.float64)  # Ensure high precision
        
        # Accumulate the samples directly
        sample_accumulator += sample_to_add
        n_points_added += 1

    # Normalize the accumulated samples by the total number of points added
    if n_points_added > 0:  # Avoid division by zero
        mbr_trajectory = sample_accumulator / n_points_added
    else:
        # Handle case with no valid points added
        mbr_trajectory = np.zeros(forward_sampling, dtype=np.float64)  # or an appropriate default/fallback value

    if return_n_points_added:
        return mbr_trajectory, n_points_added
    else:
        return mbr_trajectory
    
def mean_back_relaxation_multiple_trajectories(dt, trajectories, Tao, timespan, multiplier=1, d_sigma_estimation_samples = 1000, return_n_points_added=False):
    mbr_list = []
    n_points_added_list = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(mean_back_realxation, dt, trajectory, Tao, timespan, multiplier, d_sigma_estimation_samples, True, False) for trajectory in trajectories]
        for future in tqdm(futures, total=len(futures)):
            mbr, n_points_added = future.result()
            mbr_list.append(mbr)
            n_points_added_list.append(n_points_added)
    
    mbr_weighted_list = [n_point*mbr/(sum(n_points_added_list)) for mbr, n_point in zip(mbr_list, n_points_added_list)]
    average_mbr = sum(mbr_weighted_list)

    if return_n_points_added:
        return average_mbr, sum(n_points_added_list)
    else:
        return average_mbr


def find_idx_of_nearest_value(array, value):
    assert array.ndim == 1, "Array must be 1D"
    i = 0
    j = 0
    nearest_idx_list = []
    while i < len(array):
        current_tao = array[i] - array[j]
        next_tao = array[i] - array[j+1] if j+1 < len(array) else float('inf')  # Avoid IndexError
        if i == j or abs(current_tao - value) <= abs(next_tao - value):  # Use <= to handle the case when they are equal
            nearest_idx_list.append(j)
            i += 1
            if i < len(array):  # Check if i is within bounds after incrementing
                current_tao = array[i] - array[j]
                next_tao = array[i] - array[j+1] if j+1 < len(array) else float('inf')
        else:
            current_tao = next_tao
            j += 1
            next_tao = array[i] - array[j+1] if j+1 < len(array) else float('inf')
    return nearest_idx_list
def find_idx_of_nearest_value_right(array, value):
    # Reverse the array and adjust the indices after calling the original function
    reversed_array = array[::-1]
    reversed_indices = find_idx_of_nearest_value(reversed_array, value)
    # Calculate the original indices based on the length of the array
    original_indices = [len(array) - 1 - idx for idx in reversed_indices]
    return original_indices
def mean_back_relaxation_with_timestamps(timestamps, trajectory, Tao, timespan, multiplier=1, d_sigma_estimation_samples = 1000, return_n_points_added=False, progress_bar = True, n_bins = 100, tao_tollerance_factor = 2):
    if isinstance(trajectory, list):
        trajectory = np.array(trajectory, dtype=np.float64)  # Ensure high precision
    if isinstance(timestamps, list):
        timestamps = np.array(timestamps, dtype=np.float64)
        
    back_idx = find_idx_of_nearest_value(timestamps, Tao)
    front_idx = find_idx_of_nearest_value_right(timestamps, timespan)

    b_array = trajectory - trajectory[back_idx]
    tao_array = timestamps - timestamps[back_idx]

    d_sample = np.random.choice(b_array, size=min(d_sigma_estimation_samples, len(b_array)), replace=False)
    d_sigma = np.std(d_sample)
    d_cutoff = d_sigma * multiplier

    mbr_bin_boundaries = np.linspace(0, timespan, n_bins)
    mbr_time = (mbr_bin_boundaries[1:] + mbr_bin_boundaries[:-1]) / 2

    valid_b = np.abs(b_array) > d_cutoff
    #valid_tao = Tao/ tao_tollerance_factor < tao_array < Tao * tao_tollerance_factor
    valid_tao = np.logical_and(Tao/ tao_tollerance_factor < tao_array, tao_array < Tao * tao_tollerance_factor)
    # Identify indices where valid_b is True
    valid_indices = np.where(valid_b & valid_tao)[0]

    # Initialize arrays to store results
    # Assuming max_bin_index is known or can be precomputed
    mbr_trajectorie = np.zeros((len(mbr_bin_boundaries) - 1,))
    mbr_n_data_points = np.zeros((len(mbr_bin_boundaries) - 1,))

    # Loop over valid indices only
    for i in tqdm(valid_indices, disable=not progress_bar, total=len(valid_indices)):
        range_forward = front_idx[i]
        sample = trajectory[i:range_forward] - trajectory[i]
        t_sample = timestamps[i:range_forward] - timestamps[i]

        b_mbr = b_array[i]
        sample_to_add = (-sample / b_mbr).astype(np.float64)

        # Bin mapping for the entire sample
        bin_mapping = np.digitize(t_sample, mbr_bin_boundaries) - 1

        # Avoid adding to the last bin if not applicable
        valid_bins = bin_mapping != len(mbr_bin_boundaries) - 1

        # Use numpy to sum and count efficiently
        np.add.at(mbr_trajectorie, bin_mapping[valid_bins], sample_to_add[valid_bins])
        np.add.at(mbr_n_data_points, bin_mapping[valid_bins], 1)

    # Normalize the accumulated samples, ensuring no division by zero
    mbr_trajectorie = np.divide(mbr_trajectorie, mbr_n_data_points, where=mbr_n_data_points != 0)

    if return_n_points_added:
        return mbr_time, mbr_trajectorie, mbr_n_data_points
    return mbr_time, mbr_trajectorie

def mean_back_relaxation_multiple_trajectories_with_timestamps(timestamps_list, trajectory_list, Tao, timespan, multiplier=1, d_sigma_estimation_samples = 1000, return_n_points_added=False, progress_bar = True, n_bins = 100, tao_tollerance_factor = 2):

    mbr_list = []
    mbr_n_data_points_list = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(mean_back_relaxation_with_timestamps, timestamps, trajectory, Tao, timespan, multiplier, d_sigma_estimation_samples, True, False, n_bins, tao_tollerance_factor) for timestamps, trajectory in zip(timestamps_list, trajectory_list)]
        for future in tqdm(futures, total=len(futures)):
            mbr_time, mbr, mbr_n_data_points = future.result()
            mbr_list.append(mbr)
            mbr_n_data_points_list.append(mbr_n_data_points)
    
    average_mbr = []
    for i in range(len(mbr_time)):
        mbr_weights = [mbr_n_data_points[i] for mbr_n_data_points in mbr_n_data_points_list]
        mbr_values = [mbr[i] for mbr in mbr_list]
        mbr_weights = np.array(mbr_weights)
        mbr_values = np.array(mbr_values)
        mbr_weighted = sum(mbr_weights*mbr_values)/sum(mbr_weights)
        average_mbr.append(mbr_weighted)
    

    return mbr_time, average_mbr


    



def analytical_mean_back_realxation(dt,time,k,Diffusion_particle,Diffusion_oscillator):
    time_list=np.linspace(0,time,int(time/dt))
    prefactor=1/2*(1-Diffusion_oscillator/Diffusion_particle)
    analytical_mbr=[prefactor*(1-np.exp(-k*t)) for t in time_list]
    return analytical_mbr
