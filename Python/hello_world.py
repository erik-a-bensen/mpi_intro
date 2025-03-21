from mpi4py import MPI
import platform

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"Hello, World! I am process {rank} of {size}. Running on {platform.node()}")