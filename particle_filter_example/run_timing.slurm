#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ebensen
#SBATCH -n 256
#SBATCH -N 2
#SBATCH --ntasks-per-node=128
#SBATCH -t 2:00:00

module purge
module load anaconda3/2022.10
conda activate mpi

cd /jet/home/ebensen/repos/mpi_intro/particle_filter_example

for nprocs in 1 2 4 8 16 32 64 128 256
do
    mpiexec -n $nprocs python timing.py
done