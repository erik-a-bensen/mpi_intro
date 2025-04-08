from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def risky_function(rank, n_iters):
    for i in range(n_iters):
        time.sleep(1)
        print(f"Running risky function iteration {i} on process {rank}.")
        if rank == 0 and i == 1:
            raise ValueError("Risky function failed catastrophyically!!!!!")

def main():
    n_iters = 3
    risky_function(rank, n_iters)
    print(f"Process {rank} waiting for all processes to finish.")
    comm.Barrier()
    print(f"Process {rank} completed execution.")

if __name__ == "__main__":
    main()