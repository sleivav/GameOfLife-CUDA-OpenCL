import numpy as np
import pycuda.driver as drv
import math
import pycuda.autoinit
from pycuda.compiler import SourceModule

from src.gol.gol import GameOfLife


class CudaGameOfLife(GameOfLife):
    def __init__(self, input_file: str, program_file: str, block_size=32):
        super().__init__(input_file, program_file)
        # Flatten data to work on kernel
        self.data_matrix = self.data_matrix.flatten()
        self.result_matrix = self.result_matrix.flatten()
        self.kernel = SourceModule(self.program_file.read())
        self.block_size = block_size

    def iterate(self, iterations: int):
        # Prepare and copy data to device
        data = self.data_matrix.astype(np.int32)
        data_gpu = drv.to_device(data)
        res_gpu = drv.to_device(data)
        dimensions = np.array([self.width, self.height])
        dims = drv.to_device(dimensions)

        # Load function and iterate
        func = self.kernel.get_function("simpleLifeKernel")

        grid_dim = math.ceil(self.width / self.block_size)
        global_size = (grid_dim * self.block_size, grid_dim * self.block_size)

        for i in range(iterations):
            func(data_gpu, dims, res_gpu,
                 block=(self.block_size, 1, 1), grid=(grid_dim, 1))
            data_gpu = res_gpu

        self.data_matrix = drv.from_device(data_gpu,
                                           (self.width, self.height), 'int')
