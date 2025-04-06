#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ebensen
#SBATCH -n 256
#SBATCH -N 2
#SBATCH --ntasks-per-node=128
#SBATCH -t 2:00:00

cd /jet/home/ebensen/repos/mpi_intro/Python 

for nprocs in 1 2 4 8 16 32 64 128 256
do
    /jet/home/ebensen/.conda/envs/mpi/bin/mpiexec -n $nprocs /jet/home/ebensen/.conda/envs/mpi/bin/python timing.py
done