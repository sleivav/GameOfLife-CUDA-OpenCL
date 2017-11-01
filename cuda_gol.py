import pycuda.driver as drv
import pycuda.autoinit
import numpy as np
import time

import data_manager

from pycuda.compiler import SourceModule

kernel = SourceModule("""
__global__ void simpleLifeKernel(int* lifeData, int worldWidth,
    int worldHeight, int* resultLifeData) {
  int worldSize = worldWidth * worldHeight;
 
  for (int cellId = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
      cellId < worldSize;
      cellId += blockDim.x * gridDim.x) {
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
""")

width = 1024
height = 1024

data_matrix = data_manager.load_data().flatten()
result_matrix = np.empty(shape=(width, height)).flatten()


def iterate(iterations: int):
    global data_matrix
    global result_matrix
    data = data_matrix.astype(np.int32)
    res = data_matrix.astype(np.int32)
    data_gpu = drv.to_device(data)
    res_gpu = drv.to_device(res)
    func = kernel.get_function("simpleLifeKernel")
    for i in range(iterations):
        func(data_gpu, np.int32(width), np.int32(height), res_gpu, block=(32, 32, 1))
        temp = data_gpu
        data_gpu = res_gpu
        res_gpu = temp
    data_matrix = drv.from_device(data_gpu, (width, height), 'int')

then = time.time()
iterate(200)
print(time.time() - then)
