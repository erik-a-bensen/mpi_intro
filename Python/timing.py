from mpi4py import MPI
import numpy as np
import os
from particle_filter_animation import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main():
    start = MPI.Wtime()
    run_particle_filter(N=100000)
    end = MPI.Wtime()
    elapsed_time = end - start

    times = np.zeros(size)
    times = comm.gather(elapsed_time, root=0)
    if rank == 0:
        if not os.path.exists("timing_results"):
            os.mkdir("timing_results")
        np.save(f"timing_results/timing_{size}_processes.npy", times)

if __name__ == "__main__":
    main()