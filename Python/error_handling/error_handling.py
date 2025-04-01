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
        if rank == 0 and i == 1:
            raise ValueError("Process 0 encountered an error.")

def main():
    n_iters = 5
    try:
        risky_function(rank, n_iters)
    except ValueError as e:
        print(f"Process {rank} encountered an error:")
        print(traceback.format_exc())
        comm.Abort()
    
    print(f"Process {rank} completed execution.")

if __name__ == "__main__":
    main()