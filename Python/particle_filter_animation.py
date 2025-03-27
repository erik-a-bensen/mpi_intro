from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm
import matplotlib.cm as cm

# Initialize MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define the model parameters
def f(x, dt=1.0):
    """State transition function: x_t = f(x_{t-1})"""
    # Simple linear model (kept simple despite complex data)
    return x + dt

def h(x, obs_var=1.0):
    """Observation function with noise: z_t = h(x_t) + v_t"""
    # Always add noise to the observation
    return x + np.random.normal(0, np.sqrt(obs_var))

def generate_data(T=100, obs_var=1.0, x0=0.0):
    """Generate synthetic data with a complex but DETERMINISTIC trajectory"""
    x_true = np.zeros(T)
    z = np.zeros(T)
    
    # Initial state
    x_true[0] = x0
    
    # Parameters for complex trajectory
    sin_amplitude = 3.0  # Amplitude of sinusoidal component
    sin_frequency = 0.5  # Frequency of sinusoidal component
    
    # Time points for direction changes (as fractions of T)
    change_points = [0.25, 0.5, 0.75]
    change_point_indices = [int(cp * T) for cp in change_points]
    
    # Fixed velocities for different segments (no randomness)
    velocities = [0.8, -1.5, 1.2, -0.5]  # Initial velocity and velocities after each change point
    current_vel_idx = 0
    velocity = velocities[current_vel_idx]
    
    # Generate initial observation
    z[0] = h(x_true[0], obs_var)
    
    for t in range(1, T):
        # Check if we're at a direction change point
        if t in change_point_indices:
            current_vel_idx += 1
            velocity = velocities[current_vel_idx]
        
        # Calculate next position deterministically with:
        # 1. Previous position
        # 2. Linear trend (with current velocity)
        # 3. Sinusoidal component
        # 4. Polynomial term
        # 5. Cube root oscillating term
        x_true[t] = (x_true[t-1] + 
                     velocity +  # Linear trend with changing velocity
                     sin_amplitude * np.sin(sin_frequency * t) +  # Sinusoidal component
                     0.05 * np.sqrt(t) * np.cos(0.3 * t) +  # Non-linear oscillating term
                     0.01 * (t/10)**2 * np.sin(0.05 * t))  # Polynomial growth with oscillation
        
        # Generate observation with noise directly in the observation function
        z[t] = h(x_true[t], obs_var)
    
    return x_true, z

def run_particle_filter():
    # Set the parameters
    T = 100  # Time steps
    N = 500  # Total number of particles
    process_variance = 3  # Process variance for the model
    observation_variance = 3.0  # Observation noise variance
    
    # Calculate particles per process (assuming N is divisible by size)
    n_local = N // size

    # Generate or load data (only on rank 0)
    if rank == 0:
        x_true, observations = generate_data(T, observation_variance)
    else:
        observations = None
        x_true = None

    # Broadcast observations to all processes
    # Using lowercase bcast which directly returns the data
    observations = comm.bcast(observations, root=0)

    # Initialize particles on each process
    local_particles = np.random.uniform(
        observations[0] - 3, observations[0] + 3, n_local
    ).tolist()  # Convert to Python list for easier handling with lowercase MPI

    # Arrays for storing results (only on rank 0)
    if rank == 0:
        mean_estimates = np.zeros(T)
        mean_estimates[0] = observations[0]
        
        # Track resampling events
        resampling_events = np.zeros(T, dtype=bool)
        
        # Instead of a single array, store particles by process
        # Shape: (time_steps, num_processes, particles_per_process)
        all_particles_history = np.zeros((T, size, n_local))
        
        # Store confidence bounds
        # Each time step will have a lower and upper bound for 95% confidence
        confidence_bounds = np.zeros((T, 2))
        
        # Initial particles
        for p in range(size):
            all_particles_history[0, p] = np.random.uniform(
                observations[0] - 3, observations[0] + 3, n_local)
        
        # Calculate initial confidence bounds
        all_initial_particles = all_particles_history[0].flatten()
        confidence_bounds[0, 0] = np.percentile(all_initial_particles, 2.5)  # Lower bound
        confidence_bounds[0, 1] = np.percentile(all_initial_particles, 97.5)  # Upper bound

    # Main particle filter loop
    for t in range(1, T):
        # 1. Prediction step - advance particles according to process model
        local_particles = [f(p) + np.random.normal(0, np.sqrt(process_variance)) for p in local_particles]
        
        # 2. Update step - compute weights based on observation likelihood
        # Use particles directly for weight calculation
        local_weights = [norm.pdf(observations[t], loc=p, scale=np.sqrt(observation_variance)) for p in local_particles]
        
        # Calculate sum of weights across all processes
        # Lowercase allreduce directly returns the result
        global_weight_sum = comm.allreduce(sum(local_weights), op=MPI.SUM)
        
        # Normalize weights
        if global_weight_sum > 0:
            local_weights = [w / global_weight_sum for w in local_weights]
        else:
            # Avoid division by zero - reset weights to uniform
            local_weights = [1.0 / (n_local * size)] * n_local
        
        # Calculate local weighted mean
        local_weighted_sum = sum(p * w for p, w in zip(local_particles, local_weights))
        
        # Get global weighted mean
        # Lowercase allreduce directly returns the result
        global_weighted_sum = comm.allreduce(local_weighted_sum, op=MPI.SUM)
        
        # 3. Resampling step
        # Calculate global effective sample size
        local_sum_sq_weights = sum(w**2 for w in local_weights)
        global_sum_sq_weights = comm.allreduce(local_sum_sq_weights, op=MPI.SUM)
        n_eff = 1.0 / global_sum_sq_weights if global_sum_sq_weights > 0 else float('inf')
        
        # Decide if resampling is needed
        resample_needed = (n_eff < N / 2)
        
        # Broadcast resampling decision to all processes
        # Lowercase bcast directly returns the result
        resample_needed = comm.bcast(resample_needed, root=0)
        
        # Track resampling events (only on rank 0)
        if rank == 0:
            resampling_events[t] = resample_needed
        
        if resample_needed:
            # Gather all particles and weights from all processes
            # Lowercase allgather directly returns a list of all values
            all_particles = comm.allgather(local_particles)
            all_weights = comm.allgather(local_weights)
            
            # Flatten the nested lists
            all_particles = [p for sublist in all_particles for p in sublist]
            all_weights = [w for sublist in all_weights for w in sublist]
            
            # Each process should sample a DIFFERENT subset
            np.random.seed(t * 1000 + rank)  # Unique seed based on time step and rank
            # Sample from the flattened lists
            indices = np.random.choice(N, n_local, p=all_weights)
            local_particles = [all_particles[i] for i in indices]
            
            # Add jitter to avoid particle collapse
            local_particles = [p + np.random.normal(0, np.sqrt(process_variance/10)) for p in local_particles]
        
        # Gather particles by process to rank 0 for visualization
        # Lowercase gather returns the gathered data directly
        if rank == 0:
            # gather returns a list of gathered values
            all_process_particles = comm.gather(local_particles, root=0)
            
            # Convert gathered particles for history storage
            for p in range(size):
                all_particles_history[t, p] = all_process_particles[p]
            
            mean_estimates[t] = global_weighted_sum
            
            # Calculate confidence bounds for this time step
            all_current_particles = all_particles_history[t].flatten()
            confidence_bounds[t, 0] = np.percentile(all_current_particles, 2.5)  # Lower bound
            confidence_bounds[t, 1] = np.percentile(all_current_particles, 97.5)  # Upper bound
        else:
            # Non-root processes just send their particles
            comm.gather(local_particles, root=0)

    # Return results (only useful on rank 0)
    if rank == 0:
        return x_true, observations, mean_estimates, all_particles_history, resampling_events, confidence_bounds
    else:
        return None, None, None, None, None, None

# Run the particle filter
x_true, observations, mean_estimates, all_particles_history, resampling_events, confidence_bounds = run_particle_filter()

# Create animation on rank 0
if rank == 0 and x_true is not None:
    fig = plt.figure(figsize=(12, 8))
    
    # Create main axis for the particle filter plot
    ax = fig.add_subplot(111)
    
    # Set up plot elements
    true_line, = ax.plot([], [], 'r-', linewidth=2, label='True State')
    obs_points, = ax.plot([], [], 'kx', alpha=0.5, label='Noisy Observations')
    est_line, = ax.plot([], [], 'b-', label='PF Estimate')
    
    # Lines for confidence bounds
    lower_bound, = ax.plot([], [], 'b--', alpha=0.5)
    upper_bound, = ax.plot([], [], 'b--', alpha=0.5)
    
    # Create a separate plot element for each process's particles
    # Use a different color for each process
    colors = cm.rainbow(np.linspace(0, 1, size))
    particle_plots = []
    
    for p in range(size):
        proc_particles, = ax.plot([], [], '.', color=colors[p], alpha=0.2, markersize=2, 
                              label=f'Process {p} Particles')
        particle_plots.append(proc_particles)
    
    # Set proper axis limits based on data
    ax.set_xlim(0, len(x_true)-1)
    y_min = min(min(x_true), min(observations), np.min(confidence_bounds[:, 0])) - 5
    y_max = max(max(x_true), max(observations), np.max(confidence_bounds[:, 1])) + 5
    ax.set_ylim(y_min, y_max)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('State')
    ax.set_title('1D Particle Tracking Filter using {} MPI Processes'.format(size))
    
    # Legend with smaller font size to save space
    ax.legend(loc='upper left', fontsize='small')
    
    # Text display for current time step
    time_text = ax.text(0.25, 0.95, '', transform=ax.transAxes)
    
    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Create list to store resampling tick markers
    resample_ticks = []
    
    # Create list to store confidence fill between objects
    confidence_fills = []
    
    def init():
        true_line.set_data([], [])
        obs_points.set_data([], [])
        est_line.set_data([], [])
        lower_bound.set_data([], [])
        upper_bound.set_data([], [])
        
        # Clear any existing confidence fills
        for fill in confidence_fills:
            fill.remove()
        confidence_fills.clear()
        
        for pp in particle_plots:
            pp.set_data([], [])
            
        # Remove any existing resampling ticks
        for tick in resample_ticks:
            tick.remove()
        resample_ticks.clear()
        
        time_text.set_text('')
        return (true_line, obs_points, est_line, lower_bound, upper_bound, *particle_plots, time_text)
    
    def animate(i):
        # Show ground truth trajectory only up to current time step (animated)
        true_line.set_data(range(i+1), x_true[:i+1])
        
        # Show observations and estimates up to current time
        obs_points.set_data(range(i+1), observations[:i+1])
        est_line.set_data(range(i+1), mean_estimates[:i+1])
        
        # Update confidence bound lines
        times = np.arange(i+1)
        lower_bound.set_data(times, confidence_bounds[:i+1, 0])
        upper_bound.set_data(times, confidence_bounds[:i+1, 1])
        
        # Clear any existing confidence fills
        for fill in confidence_fills:
            fill.remove()
        confidence_fills.clear()
        
        # Add new confidence region
        if len(times) > 1:  # Only add if we have at least 2 points
            fill = ax.fill_between(times, 
                                confidence_bounds[:i+1, 0],
                                confidence_bounds[:i+1, 1],
                                alpha=0.2, color='blue')
            confidence_fills.append(fill)
        
        # Update resampling tick markers
        # First, remove any existing ticks
        for tick in resample_ticks:
            tick.remove()
        resample_ticks.clear()
        
        # Find resampling events up to current time
        resample_times = np.where(resampling_events[:i+1])[0]
        
        # Add vertical lines at each resampling event
        for t in resample_times:
            line = ax.axvline(x=t, ymin=0, ymax=0.025, color='m', linewidth=2)
            resample_ticks.append(line)
            
        # Display particles from each process with different colors
        for p in range(size):
            # Get particles for this process at the current time step
            proc_particles = all_particles_history[i, p]
            # Draw them at the current time step with their process color
            particle_plots[p].set_data([i] * len(proc_particles), proc_particles)
        
        # Update time step text
        time_text.set_text(f'Time Step: {i}')
        
        # Create return tuple with all elements that might change
        result = (true_line, obs_points, est_line, lower_bound, upper_bound, 
                 *particle_plots, time_text, *resample_ticks)
        
        # Add confidence fills to return tuple if they exist
        if confidence_fills:
            result = result + tuple(confidence_fills)
            
        return result
    
    # Parameters for a smoother animation
    T = len(x_true)
    
    # Add a text annotation explaining the magenta ticks
    plt.figtext(0.5, 0.15, 'Resampling events', 
                ha='center', color='magenta', fontweight='bold')
    
    # Set blit=False to avoid issues with collections
    anim = animation.FuncAnimation(fig, animate, frames=T,
                                  init_func=init, blit=False, interval=150)
    
    # Save animation
    try:
        # Check if ffmpeg is available
        writer = animation.writers['ffmpeg'](fps=10)
        anim.save('particle_filter.mp4', writer=writer)
        print("Animation saved as 'particle_filter_with_confidence.mp4'")
    except Exception as e:
        print(f"Could not save animation: {e}")
        print("Displaying animation instead...")
    
    plt.show()