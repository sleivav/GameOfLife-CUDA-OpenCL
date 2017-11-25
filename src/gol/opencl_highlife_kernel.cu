__kernel void simpleLifeKernel(__global int* data, __global int* dims, __global int* res) {
  int worldWidth = dims[0];
  int worldHeight = dims[1];
  int worldSize = worldWidth * worldHeight;

  for (int cellId = get_global_id(0);
      cellId < worldSize;
      cellId += get_local_size(0) * get_num_groups(0)) {
    int x = cellId % worldWidth;
    int y = cellId - x;
    int xLeft = (x + worldWidth - 1) % worldWidth;
    int xRight = (x + 1) % worldWidth;
    int yUp = (y + worldSize - worldWidth) % worldSize;
    int yDown = (y + worldWidth) % worldSize;

    int aliveCells = data[xLeft + yUp] + data[x + yUp] +
                     data[xRight + yUp] + data[xLeft + y] + data[xRight + y] +
                     data[xLeft + yDown] + data[x + yDown] + data[xRight + yDown];

    res[x + y] = (!data[x+y] && (aliveCells == 3 || aliveCells == 6) ||
                 (data[x+y] && (aliveCells == 2 || aliveCells == 3 )) ? 1 : 0;
  }
}
