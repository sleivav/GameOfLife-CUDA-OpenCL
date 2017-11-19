import os
import unittest

import numpy as np
from src.gol.cuda_gol import CudaGameOfLife
from src.gol.opencl_gol import OpenCLGameOfLife
from src.gol.sequential_gol import SequentialGameOfLife

from src.gol.data_manager import DataManager


class ConsistencyTest(unittest.TestCase):
    def testCuda(self):
        test_file = 'test.out'

        DataManager.generate_data(5, 5, 0.5, test_file)

        src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
        cuda_file = os.path.join(src_path, 'cuda_gol_kernel.cu')
        opencl_file = os.path.join(src_path, 'opencl_gol_kernel.cl')

        cudaGol_small = CudaGameOfLife(test_file, program_file=str(cuda_file))
        openClGol_small = OpenCLGameOfLife(test_file, program_file=str(opencl_file))
        sequential_small = SequentialGameOfLife(test_file)

        sequential_small.iterate(30)
        openClGol_small.iterate(30)
        cudaGol_small.iterate(30)
        #cudaGol_small.iterate(30)
        #openClGol_small.iterate(30)
        np.testing.assert_array_equal(sequential_small.data_matrix.flatten(),
                                      cudaGol_small.data_matrix.flatten())
        np.testing.assert_array_equal(sequential_small.data_matrix.flatten(),
                                      openClGol_small.data_matrix.flatten())


if __name__ == '__main__':
    unittest.main()
