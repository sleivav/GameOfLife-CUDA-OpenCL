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

    int counter = 0;

    if(data[xLeft + yUp]) {
        counter++;
    }
    if(data[x + yUp]) {
        counter++;
    }
    if(data[xRight + yUp]) {
        counter++;
    }
    if(data[xLeft + y]) {
        counter++;
    }
    if(data[xRight + y]) {
        counter++;
    }
    if(data[xLeft + yDown]) {
        counter++;
    }
    if(data[x + yDown]) {
        counter++;
    }
    if(data[xRight + yDown]) {
        counter++;
    }

    if(counter == 3 || (counter == 2 && data[x + y])) {
        res[x + y] = 1;
    } else {
        res[x + y] = 0;
    }
  }
}
