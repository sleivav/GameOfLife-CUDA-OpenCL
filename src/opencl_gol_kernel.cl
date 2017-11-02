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
