from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
import os

from data_manager import DataManager

width = 5
height = 5
dimensions = np.array([width, height])

data_matrix = DataManager.load_data('test.out').flatten()
result_matrix = np.empty(shape=(width, height)).flatten()

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

prg = cl.Program(ctx, """
__kernel void simpleLifeKernel(__global int* lifeData, __global int* dims, __global int* resultLifeData) {
  int worldWidth = dims[0];
  int worldHeight = dims[1];
  int worldSize = worldWidth * worldHeight;
 
  for (int cellId = get_global_id(0);
      cellId < worldSize;
      cellId += get_local_size(0) * get_num_groups(0)) {
    int x = cellId % worldWidth;
    int yAbs = cellId - x;
    int xLeft = (x + worldWidth - 1) % worldWidth;
    int xRight = (x + 1) % worldWidth;
    int yAbsUp = (yAbs + worldSize - worldWidth) % worldSize;
    int yAbsDown = (yAbs + worldWidth) % worldSize;
 
    int aliveCells = lifeData[xLeft + yAbsUp] + lifeData[x + yAbsUp]
      + lifeData[xRight + yAbsUp] + lifeData[xLeft + yAbs] + lifeData[xRight + yAbs]
      + lifeData[xLeft + yAbsDown] + lifeData[x + yAbsDown] + lifeData[xRight + yAbsDown];
 
    resultLifeData[x + yAbs] =
      aliveCells == 3 || (aliveCells == 2 && lifeData[x + yAbs]) ? 1 : 0;
  }
}
""").build()


def iterate(iterations: int):
    global data_matrix
    global result_matrix
    data = data_matrix.astype(np.int32)
    res = data_matrix.astype(np.int32)
    life_data = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=data)
    result_life_data = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=res)
    dim = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dimensions)
    for i in range(iterations):
        prg.simpleLifeKernel(queue, data_matrix.shape, None,
                             life_data, dim, result_life_data)
        life_data = result_life_data
    res_np = np.empty_like(res)
    cl.enqueue_copy(queue, res_np, result_life_data).wait()
    data_matrix = res_np

iterate(1)
print(data_matrix)