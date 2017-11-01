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