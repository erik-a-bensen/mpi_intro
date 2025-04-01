from mpi4py import MPI
import numpy as np
import time
from collections import deque

# Initialize MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def process_task(task):
    """Simulate processing a task by sleeping for a random time"""
    task_id, work_amount = task
    time.sleep(work_amount)  # Simulate work by sleeping
    return (task_id, f"Task {task_id} completed by process {rank}")

# Main process (rank 0) creates and manages tasks
if rank == 0:
    print(f"Running with {size} processes")
    
    # Create a queue of tasks
    num_tasks = (size - 1) * 10  # 3 tasks per worker
    task_queue = deque()
    for i in range(num_tasks):
        # Task is a tuple of (task_id, work_amount)
        task_queue.append((i, np.random.random()))  # Random work between 0-1 seconds
    
    # Track which workers are busy (indexed by rank)
    busy_workers = [False] * size  # rank 0 is never busy as it's the manager
    results = []
    
    # Initial distribution of tasks
    for worker in range(1, size):
        if task_queue:
            task = task_queue.popleft()
            comm.send(task, dest=worker)
            busy_workers[worker] = True
            print(f"Sent initial task {task[0]} to worker {worker}")
    
    # Process until all tasks are done
    while task_queue or any(busy_workers[1:]):
        # Check for completed work from any worker
        status = MPI.Status()
        result = comm.recv(source=MPI.ANY_SOURCE, status=status)
        worker_rank = status.Get_source()
        
        # Mark the worker as available
        busy_workers[worker_rank] = False
        results.append(result)
        print(f"Received result from worker {worker_rank}: {result}")
        
        # Assign a new task to this worker if available
        if task_queue:
            task = task_queue.popleft()
            comm.send(task, dest=worker_rank)
            busy_workers[worker_rank] = True
            print(f"Sent task {task[0]} to worker {worker_rank}")
    
    # Out of tasks, tell all workers to exit
    for worker in range(1, size):
        comm.send(None, dest=worker)

# Worker processes
else:
    while True:
        # Receive a task
        task = comm.recv(source=0)
        
        # Exit if received termination signal
        if task is None:
            break
        
        # Process the task and send back result
        result = process_task(task)
        comm.send(result, dest=0)
    
    print(f"Worker {rank} exiting")

# Ensure all processes finish
comm.Barrier()
if rank == 0:
    print("All processes have finished.")
    completed_tasks = [[] for _ in range(size-1)]
    for task in results:
        task_id, message = task
        worker_rank = int(message.split()[-1])  # Extract worker rank from message
        completed_tasks[worker_rank - 1].append(task_id)
    print("Completed tasks by each worker:")
    for i, tasks in enumerate(completed_tasks):
        print(f"Worker {i + 1}: {tasks}")