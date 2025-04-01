from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm
import matplotlib.cm as cm

# Define the model parameters
def f(x, dt=1.0):
    """State transition function: x_t = f(x_{t-1})"""
    # Simple linear model (kept simple despite complex data)
    return x + dt

def h(x, obs_var=1.0):
    """Observation function with noise: z_t = h(x_t) + v_t"""
    # Always add noise to the observation
    return x + np.random.normal(0, np.sqrt(obs_var))

def get_measurement(observations, t):
    """Get the measurement at time t."""
    return observations[t]

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

def initialize_particles(observation, n_particles, jitter=3):
    """Initialize particles based on first observation."""
    return np.random.uniform(observation - jitter, observation + jitter, n_particles).tolist()

def predict_particles(particles, process_variance):
    """Move particles according to process model."""
    return [f(p) + np.random.normal(0, np.sqrt(process_variance)) for p in particles]

def calculate_weights(particles, observation, observation_variance):
    """Calculate weights based on observation likelihood."""
    return [norm.pdf(observation, loc=p, scale=np.sqrt(observation_variance)) for p in particles]

def update_particles(local_particles, process_variance, observations, observation_variance):
    local_particles = predict_particles(local_particles, process_variance)
    local_weights = calculate_weights(local_particles, observations, observation_variance)
    return local_particles, local_weights

def calculate_effective_sample_size(weights):
    """Calculate the effective sample size based on weights."""
    sum_sq_weights = sum(w**2 for w in weights)
    return 1.0 / sum_sq_weights if sum_sq_weights > 0 else float('inf')

def resample_particles(all_particles, all_weights, n_local, seed, process_variance):
    """Resample particles based on weights and add jitter."""
    np.random.seed(seed)
    indices = np.random.choice(len(all_particles), n_local, p=all_weights)
    resampled_particles = [all_particles[i] for i in indices]
    # Add jitter to avoid particle collapse
    return [p + np.random.normal(0, np.sqrt(process_variance/10)) for p in resampled_particles]

def initialize_result_arrays(T, size, n_local, first_observation):
    """Initialize arrays for storing results."""
    mean_estimates = np.zeros(T)
    mean_estimates[0] = first_observation
    
    resampling_events = np.zeros(T, dtype=bool)
    
    # Shape: (time_steps, num_processes, particles_per_process)
    all_particles_history = np.zeros((T, size, n_local))
    
    # Each time step will have a lower and upper bound for 95% confidence
    confidence_bounds = np.zeros((T, 2))
    
    # Initialize particles for all processes
    for p in range(size):
        all_particles_history[0, p] = initialize_particles(first_observation, n_local)
    
    # Calculate initial confidence bounds
    all_initial_particles = all_particles_history[0].flatten()
    confidence_bounds[0, 0] = np.percentile(all_initial_particles, 2.5)  # Lower bound
    confidence_bounds[0, 1] = np.percentile(all_initial_particles, 97.5)  # Upper bound
    
    return mean_estimates, resampling_events, all_particles_history, confidence_bounds

def update_results(t, all_process_particles, global_weighted_sum, mean_estimates, 
                  all_particles_history, confidence_bounds, size):
    """Update results arrays with new data."""
    # Store gathered particles
    for p in range(size):
        all_particles_history[t, p] = all_process_particles[p]
    
    # Update mean estimate
    mean_estimates[t] = global_weighted_sum
    
    # Calculate confidence bounds for this time step
    all_current_particles = all_particles_history[t].flatten()
    confidence_bounds[t, 0] = np.percentile(all_current_particles, 2.5)  # Lower bound
    confidence_bounds[t, 1] = np.percentile(all_current_particles, 97.5)  # Upper bound

def build_animation(results, size):
    x_true, observations, mean_estimates, all_particles_history, resampling_events, confidence_bounds = results

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