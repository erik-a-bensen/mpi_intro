from mpi4py import MPI
import numpy as np
import os
from particle_filter_animation import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main():
    start = MPI.Wtime()
    run_particle_filter()
    end = MPI.Wtime()
    elapsed_time = end - start

    times = np.zeros(size)
    comm.Gather(elapsed_time, times, root=0)
    if rank == 0:
        os.mkdir("timing_results", exist_ok=True)
        np.save(f"timing_results/timing_{size}_processes.npy", times)

if __name__ == "__main__":
    main()