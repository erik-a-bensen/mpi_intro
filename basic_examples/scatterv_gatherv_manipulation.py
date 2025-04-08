from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def run_scatterv_example():
    start_time = time.time()
    
    # Fixed total of 107 temperature readings (in Fahrenheit)
    total_points = 107
    
    # Calculate points per process (as evenly as possible)
    # Each process gets base_count, and the first remainder processes get one extra
    base_count = total_points // size
    remainder = total_points % size
    
    # Calculate how many points each process gets
    points_per_process = np.array([base_count + (1 if i < remainder else 0) for i in range(size)], dtype=np.int32)
    
    # Calculate displacements for scatterv
    displacements = np.zeros(size, dtype=np.int32)
    for i in range(1, size):
        displacements[i] = displacements[i-1] + points_per_process[i-1]
    
    # Generate sample temperature data on root process (in Fahrenheit)
    if rank == 0:
        # Create temperature data (68°F to 86°F range)
        all_temps_f = np.array([68 + 0.2*i for i in range(total_points)], dtype=np.float64)
        print(f"Temperature Conversion Example (107 total readings):")
        print(f"  Number of processes: {size}")
        print(f"  Readings per process: {points_per_process}")
        print(f"  Displacements: {displacements}")
        print(f"  Total readings: {total_points}")
        print(f"  Sample temperatures (F): {all_temps_f[:5]}...")
    else:
        all_temps_f = None
    
    # Allocate space for local temperature data (size based on distribution)
    local_temps_f = np.empty(points_per_process[rank], dtype=np.float64)
    
    # Scatter temperature data with different amounts to each process
    comm.scatterv([all_temps_f, points_per_process, displacements, MPI.DOUBLE], local_temps_f, root=0)
    
    # Each process converts its temperatures from Fahrenheit to Celsius
    # Formula: C = (F - 32) * 5/9
    local_temps_c = (local_temps_f - 32) * 5/9
    
    # Prepare to collect all converted temperatures back to root
    if rank == 0:
        all_temps_c = np.empty(total_points, dtype=np.float64)
    else:
        all_temps_c = None
    
    # Gather all converted temperatures back to root with varying counts
    comm.gatherv(local_temps_c, [all_temps_c, points_per_process, displacements, MPI.DOUBLE], root=0)
    
    # Process 0 validates and reports results
    if rank == 0:
        print(f"Scatterv/gatherv conversion results:")
        print(f"  Sample original temperatures (F): {all_temps_f[:5]}...")
        print(f"  Sample converted temperatures (C): {all_temps_c[:5]}...")
        
        # Verify the conversion is correct by comparing with direct calculation
        direct_conversion = (all_temps_f - 32) * 5/9
        max_diff = np.max(np.abs(all_temps_c - direct_conversion))
        print(f"  Maximum difference from direct calculation: {max_diff:.10f}")
        print(f"  Time: {time.time() - start_time:.6f} seconds")

if __name__ == "__main__":
    run_scatterv_example()