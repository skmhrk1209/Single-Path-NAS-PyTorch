import torch
from torch import distributed
from mpi4py import MPI
import socket
import os


def init_process_group(backend):

    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    info = dict()
    if rank == 0:
        host = socket.gethostname()
        address = socket.gethostbyname(host)
        info.update(dict(MASTER_ADDR=address, MASTER_PORT='1234'))

    info = comm.bcast(info, root=0)
    info.update(dict(WORLD_SIZE=str(world_size), RANK=str(rank)))
    os.environ.update(info)

    distributed.init_process_group(backend=backend)
