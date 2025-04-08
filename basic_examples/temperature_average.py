from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Number of data points per process (for a more realistic example)
points_per_process = 5_000_000
total_points = size * points_per_process

def generate_temps():
    # Generate temperature data (in Fahrenheit) for all processes
    np.random.seed(12)
    return np.random.uniform(32, 100, total_points).astype(np.float64)

# 1. Send/Recv Version
def run_send_recv():
    start_time = time.time()
    
    # Generate sample data on root process
    if rank == 0:
        # Create temperature data (multiple points per process)
        all_temps = generate_temps()
        
        # Send each process its chunk of data
        for dest in range(1, size):
            start_idx = dest * points_per_process
            end_idx = start_idx + points_per_process
            chunk = all_temps[start_idx:end_idx]
            comm.send(chunk, dest=dest)
        
        # Root keeps its own chunk
        local_chunk = all_temps[:points_per_process]
    else:
        # Other processes receive their chunk
        local_chunk = comm.recv(source=0)
    
    # Everyone processes their local data
    local_result = np.mean(local_chunk)
    
    # Process 0 will collect all processed values and compute average
    if rank == 0:
        # Store our own processed result
        processed_results = [local_result]
        
        # Receive processed results from all other processes
        for src in range(1, size):
            result = comm.recv(source=src)
            processed_results.append(result)
        
        # Calculate and print average
        final_avg = sum(processed_results) / len(processed_results)
        print(f"Send/Recv version:")
        print(f"  Final average: {final_avg:.4f}")
        print(f"  Time: {time.time() - start_time:.6f} seconds")
    else:
        # All other processes send their processed result to process 0
        comm.send(local_result, dest=0)

# 2. Scatter/Gather Version
def run_scatter_gather():
    start_time = time.time()
    
    # Generate sample data on root process
    if rank == 0:
        # Create temperature data (multiple points per process)
        all_temps = generate_temps()
        # Reshape into chunks for each process
        data_chunks = all_temps.reshape(size, points_per_process)
    else:
        data_chunks = None
    
    # Scatter the chunks to all processes
    local_chunk = comm.scatter(data_chunks, root=0)
    
    # Everyone processes their local data
    local_result = np.mean(local_chunk)
    
    # Gather all processed results back to root
    processed_results = comm.gather(local_result, root=0)
    
    # Process 0 computes the final average
    if rank == 0:
        final_avg = sum(processed_results) / len(processed_results)
        print(f"Scatter/Gather version:")
        print(f"  Final average: {final_avg:.4f}")
        print(f"  Time: {time.time() - start_time:.6f} seconds")

# 3. Bcast/Gather Version
def run_bcast_gather():
    start_time = time.time()
    
    # Generate sample data on root process
    if rank == 0:
        # Create temperature data (multiple points per process)
        all_temps = generate_temps()
    else:
        all_temps = np.empty(total_points, dtype=np.float64)
    
    # Broadcast all data to everyone
    comm.Bcast(all_temps, root=0)
    
    # Each process works on its assigned chunk
    start_idx = rank * points_per_process
    end_idx = start_idx + points_per_process
    local_chunk = all_temps[start_idx:end_idx]
    
    # Process the local chunk
    local_result = np.mean(local_chunk)
    
    # Gather all processed results back to root
    processed_results = comm.gather(local_result, root=0)
    
    # Process 0 computes the final average
    if rank == 0:
        final_avg = sum(processed_results) / len(processed_results)
        print(f"Bcast/Gather version:")
        print(f"  Final average: {final_avg:.4f}")
        print(f"  Time: {time.time() - start_time:.6f} seconds")

# 4. Scatter/Reduce Version
def run_scatter_reduce():
    start_time = time.time()
    
    # Generate sample data on root process
    if rank == 0:
        # Create temperature data (multiple points per process)
        all_temps = generate_temps()
        # Reshape into chunks for each process
        data_chunks = all_temps.reshape(size, points_per_process)
    else:
        data_chunks = None
    
    # Scatter the chunks to all processes
    local_chunk = comm.scatter(data_chunks, root=0)
    
    # Everyone processes their local data
    local_result = np.mean(local_chunk)
    
    # Calculate sum of all results using reduce
    total_result = comm.reduce(local_result, op=MPI.SUM, root=0)
    
    # Process 0 computes the final average
    if rank == 0:
        final_avg = total_result / size
        print(f"Scatter/Reduce version:")
        print(f"  Final average: {final_avg:.4f}")
        print(f"  Time: {time.time() - start_time:.6f} seconds")

if __name__ == "__main__":
    run_send_recv()
    run_scatter_gather()
    run_bcast_gather()
    run_scatter_reduce()