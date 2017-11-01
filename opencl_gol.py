from __future__ import absolute_import, print_function
from gol import GameOfLife
import numpy as np
import pyopencl as cl


class OpenCLGameOfLife(GameOfLife):
    def __init__(self, input_file: str, program_file=None):
        super().__init__(input_file, program_file)
        # Flatten matrices to work on kernel
        self.data_matrix = self.data_matrix.flatten()
        self.result_matrix = self.result_matrix.flatten()
        # OpenCL arguments and initialization
        self.dimensions = np.array([self.width, self.height])
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags
        self.kernel = cl.Program(self.ctx, self.program_file.read()).build()

    def iterate(self, iterations: int):
        # Prepare data to copy to device
        data = self.data_matrix.astype(np.int32)
        res = self.data_matrix.astype(np.int32)

        rwc = self.mf.READ_WRITE | self.mf.COPY_HOST_PTR
        rc = self.mf.READ_ONLY | self.mf.COPY_HOST_PTR
        # Copy data to device
        life_data = cl.Buffer(self.ctx, rwc, hostbuf=data)
        result_life_data = cl.Buffer(self.ctx, rwc, hostbuf=res)
        dim = cl.Buffer(self.ctx, rc, hostbuf=self.dimensions)

        for i in range(iterations):
            self.kernel.simpleLifeKernel(self.queue, self.data_matrix.shape, None,
                                         life_data, dim, result_life_data)
            life_data = result_life_data
        res_np = np.empty_like(res)
        cl.enqueue_copy(self.queue, res_np, result_life_data).wait()
        self.data_matrix = res_np
