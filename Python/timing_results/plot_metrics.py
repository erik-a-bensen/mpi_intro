import numpy as np 
import matplotlib.pyplot as plt 
import os 

def main():
    files = os.listdir(".")
    files = [f for f in files if f.startswith("timing_") and f.endswith(".npy")]
    files.sort()
    processes = np.array([f.split("_")[1] for f in files])
    mean_times = np.array([np.mean(np.load(f)) for f in files])
    std_times = np.array([np.std(np.load(f)) for f in files])

    fig, ax = plt.subplots()
    ax.errorbar(processes, mean_times, yerr=std_times, fmt='o')
    ax.set_xlabel("Number of Processes")
    ax.set_ylabel("Time (s)")
    ax.set_title("Particle Filter Timing Results")
    ax.set_xticks(processes)
    ax.set_xticklabels(processes)
    ax.set_xscale("log")
    ax.grid(True)
    plt.savefig("particle_filter_timing.png")

if __name__ == "__main__":
    main()