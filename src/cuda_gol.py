import numpy as np
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule

from src.gol import GameOfLife


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
        res = self.data_matrix.astype(np.int32)
        data_gpu = drv.to_device(data)
        res_gpu = drv.to_device(res)

        # Load function and iterate
        func = self.kernel.get_function("simpleLifeKernel")
        for i in range(iterations):
            func(data_gpu, np.int32(self.width),
                 np.int32(self.height), res_gpu,
                 block=(self.block_size, self.block_size, 1))
            temp = data_gpu
            data_gpu = res_gpu
            res_gpu = temp

        self.data_matrix = drv.from_device(data_gpu,
                                           (self.width, self.height), 'int')
