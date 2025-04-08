from mpi4py import MPI
import traceback
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def risky_function(rank, n_iters):
    for i in range(n_iters):
        time.sleep(1)
        print(f"Running risky function iteration {i} on process {rank}.")
        if rank == size-1 and i == 1:
            raise ValueError("Risky function failed catastrophyically!!!!!")
        
def safe_risky_function(rank, n_iters):
    try:
        risky_function(rank, n_iters)
    except ValueError as e:
        print(f"Process {rank} encountered an error:")
        print(traceback.format_exc())
        comm.Abort()
    else:
        print(f"Process {rank} completed risky function without errors.")

def main():
    n_iters = 5
    safe_risky_function(rank, n_iters)
    print(f"Process {rank} waiting for all processes to finish.")
    comm.Barrier()
    print(f"Process {rank} completed execution.")

if __name__ == "__main__":
    main()