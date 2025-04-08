#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ebensen
#SBATCH -n 256
#SBATCH -N 2
#SBATCH --ntasks-per-node=128
#SBATCH -t 0:05:00

module purge
module load anaconda3/2022.10
conda activate mpi

cd ~/repos/mpi_intro/basic_examples

mpiexec -n 256 python hello_world.py