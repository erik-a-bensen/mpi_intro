from mpi4py import MPI
import numpy as np
from particle_filter_utils import *

# Initialize MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def run_particle_filter():
    # Set parameters
    T = 100  # Time steps
    N = 100000  # Total number of particles
    process_variance = 3  # Process variance
    observation_variance = 3.0  # Observation noise variance
    
    # Calculate particles per process
    n_local = N // size
    # Store the number of particles on each process for gatherv operations
    local_counts = comm.allgather([n_local])
    
    # === DATA PREPARATION (MPI) ===
    # Generate data (only on rank 0)
    x_true, observations, observation = None, None, None
    if rank == 0:
        x_true, observations = generate_data(T, observation_variance)
        observation = observations[0]
    # Initialize single observation variable
    observation = comm.bcast(observation, root=0)
    
    # Initialize particles on each process
    local_particles = initialize_particles(observation, n_local)
    
    # Initialize result arrays (only on rank 0)
    if rank == 0:
        mean_estimates, resampling_events, all_particles_history, confidence_bounds = initialize_result_arrays(
            T, size, n_local, observations[0])
    
    # === MAIN PARTICLE FILTER LOOP ===
    for t in range(1, T):
        # If main process, get observation 
        if rank == 0:
            observation = get_measurement(observations, t)
        # Broadcast the observation to all processes
        observation = comm.bcast(observation, root=0)
        
        # Update particles based on the previous state and process noise
        local_particles, local_weights = update_particles(local_particles, process_variance, observation, observation_variance)
        
        # === MPI ===
        # Calculate sum of weights across all processes
        global_weight_sum = comm.allreduce(sum(local_weights), op=MPI.SUM)
        
        # Normalize weights & get local weighted mean
        local_weights = [w / global_weight_sum for w in local_weights]
        local_weighted_sum = sum(p * w for p, w in zip(local_particles, local_weights))
        
        # === MPI ===
        # Get global weighted mean
        global_weighted_sum = comm.allreduce(local_weighted_sum, op=MPI.SUM)
        
        # Calculate effective sample size
        local_sum_sq_weights = sum(w**2 for w in local_weights)
        
        # === MPI ===
        # Get global sum of squared weights
        global_sum_sq_weights = comm.allreduce(local_sum_sq_weights, op=MPI.SUM)
        n_eff = 1.0 / global_sum_sq_weights if global_sum_sq_weights > 0 else float('inf')
        
        # Decide if resampling is needed
        resample_needed = (n_eff < N / 2)
        
        if resample_needed:
            # MPI: Gather all particles and weights
            all_particles = comm.allgather(local_particles)
            all_weights = comm.allgather(local_weights)
            
            # Flatten the nested lists
            all_particles = [p for sublist in all_particles for p in sublist]
            all_weights = [w for sublist in all_weights for w in sublist]
            
            # Resample particles with different seed per process
            local_particles = resample_particles(
                all_particles, all_weights, n_local, t * 1000 + rank, process_variance)
        
        # === RESULTS COLLECTION (MPI) ===
        if rank == 0:
            # MPI: Gather particles from all processes
            all_process_particles = comm.gather(local_particles, root=0)
            
            # Update result arrays
            resampling_events[t] = resample_needed
            update_results(t, all_process_particles, global_weighted_sum, 
                          mean_estimates, all_particles_history, confidence_bounds, size)
        else:
            # Non-root processes just send their particles
            comm.gather(local_particles, root=0)
    
    # Return results (only useful on rank 0)
    if rank == 0:
        return x_true, observations, mean_estimates, all_particles_history, resampling_events, confidence_bounds
    else:
        return None, None, None, None, None, None

if __name__ == "__main__":
    # Run the particle filter
    results = run_particle_filter()

    # Create animation on rank 0
    if rank == 0 and results[0] is not None:
        build_animation(results, size)