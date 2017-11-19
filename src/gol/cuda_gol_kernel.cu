__global__ void simpleLifeKernel(int* data, int* dims, int* res) {
  int worldWidth = dims[0];
  int worldHeight = dims[1];
  int size = worldWidth * worldHeight;

  for (int cellId = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
       cellId < size;
       cellId += blockDim.x * gridDim.x) {
    int x = cellId % worldWidth;
    int y = cellId - x;
    int xLeft = (x + worldWidth - 1) % worldWidth;
    int xRight = (x + 1) % worldWidth;
    int yUp = (y + size - worldWidth) % size;
    int yDown = (y + worldWidth) % size;

    int aliveCells = data[xLeft + yUp] + data[x + yUp] +
                     data[xRight + yUp] + data[xLeft + y] + data[xRight + y] +
                     data[xLeft + yDown] + data[x + yDown] + data[xRight + yDown];

    res[x + y] = aliveCells == 3 || (aliveCells == 2 && data[x + y]) ? 1 : 0;
  }
}
