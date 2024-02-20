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

        


    



def analytical_mean_back_realxation(dt,time,k,Diffusion_particle,Diffusion_oscillator):
    time_list=np.linspace(0,time,int(time/dt))
    prefactor=1/2*(1-Diffusion_oscillator/Diffusion_particle)
    analytical_mbr=[prefactor*(1-np.exp(-k*t)) for t in time_list]
    return analytical_mbr
