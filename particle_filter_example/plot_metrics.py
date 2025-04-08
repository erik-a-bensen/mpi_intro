import numpy as np 
import matplotlib.pyplot as plt 
import os 

def main():
    files = os.listdir("./timing_results")
    files = [f for f in files if f.startswith("timing_") and f.endswith(".npy")]
    
    # Sort files by number of processes
    file_process_pairs = [(f, int(f.split("_")[1])) for f in files]
    file_process_pairs.sort(key=lambda x: x[1])
    files = [pair[0] for pair in file_process_pairs]
    processes = np.array([pair[1] for pair in file_process_pairs])
    
    # Load timing data
    timing_data = [np.load(os.path.join('timing_results', f)) for f in files]
    mean_times = np.array([np.mean(data) for data in timing_data])
    std_times = np.array([np.std(data) for data in timing_data])
    
    # Get sequential time (time with 1 process)
    sequential_idx = np.where(processes == 1)[0]
    if len(sequential_idx) > 0:
        sequential_time = mean_times[sequential_idx[0]]
    else:
        # Fallback if no 1-process data (should not happen with your setup)
        sequential_time = mean_times[0] * processes[0]
    
    # Calculate parallel metrics
    speedup = sequential_time / mean_times
    efficiency = speedup / processes
    
    # Estimate parallel fraction using Amdahl's Law
    # 1/S = (1-f) + f/p where S is speedup, f is parallel fraction, p is number of processors
    parallel_fraction = []
    for i, p in enumerate(processes):
        if p == 1:
            parallel_fraction.append(1.0)  # By definition
        else:
            # Solve for f: f = (p/S - 1)/(p - 1)
            s = speedup[i]
            f = (p/s - 1)/(p - 1)
            # Bound between 0 and 1
            f = max(0, min(1, 1-f))
            parallel_fraction.append(f)
    serial_fraction = 1 - np.array(parallel_fraction)
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Wall Time vs Processes (log-log)
    axs[0, 0].errorbar(processes, mean_times, yerr=std_times, fmt='o-')
    axs[0, 0].set_xlabel("Number of Processes")
    axs[0, 0].set_ylabel("Time (s)")
    axs[0, 0].set_title("Wall Time")
    axs[0, 0].set_xscale("log")
    axs[0, 0].set_yscale("log")
    # Add ideal scaling line
    ideal_times = sequential_time / processes
    axs[0, 0].plot(processes, ideal_times, 'k--', label='Ideal Scaling')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot 2: Speedup vs Processes
    axs[0, 1].plot(processes, speedup, 'o-')
    axs[0, 1].plot(processes, processes, 'k--', label='Ideal Speedup')  # Ideal speedup line
    axs[0, 1].set_xlabel("Number of Processes")
    axs[0, 1].set_ylabel("Speedup")
    axs[0, 1].set_title("Parallel Speedup")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot 3: Efficiency vs Processes
    axs[1, 0].plot(processes, efficiency, 'o-')
    axs[1, 0].axhline(y=1.0, color='k', linestyle='--', label='Ideal Efficiency')
    axs[1, 0].set_xlabel("Number of Processes")
    axs[1, 0].set_ylabel("Efficiency")
    axs[1, 0].set_title("Parallel Efficiency")
    axs[1, 0].set_ylim(0, 1.1)
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Plot 4: Estimated Parallel Fraction vs Processes
    axs[1, 1].plot(processes, serial_fraction, 'o-')
    axs[1, 1].axhline(y=0.0, color='k', linestyle='--', label='Fully Parallel')
    axs[1, 1].set_xlabel("Number of Processes")
    axs[1, 1].set_ylabel("Serial Fraction")
    axs[1, 1].set_title("Karp-Flatt Metric (Serial Fraction)")
    # axs[1, 1].set_ylim(0, 1.1)
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig("particle_filter_performance.png", dpi=300)

if __name__ == "__main__":
    main()