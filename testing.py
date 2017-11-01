from datetime import datetime

from cuda_gol import CudaGameOfLife
from opencl_gol import OpenCLGameOfLife
from sequential_gol import SequentialGameOfLife
from data_manager import DataManager


def basicTest(width: int, height: int, prob: float):
    test_file = 'test.out'

    DataManager.generate_data(width, height, prob, test_file)

    cudaGol_small = CudaGameOfLife(test_file, program_file='cuda_gol_kernel.cu')
    openClGol_small = OpenCLGameOfLife(test_file, program_file='opencl_gol_kernel.cl')
    sequential_small = SequentialGameOfLife(test_file)

    time_start = datetime.now()
    sequential_small.iterate(200)
    time_end = datetime.now()
    diff = str((time_end - time_start).total_seconds())
    print("Tiempo secuencial: " + diff)

    time_start = datetime.now()
    cudaGol_small.iterate(200)
    time_end = datetime.now()
    diff = str((time_end - time_start).total_seconds())
    print("Tiempo cuda: " + diff)

    time_start = datetime.now()
    openClGol_small.iterate(200)
    time_end = datetime.now()
    diff = str((time_end - time_start).total_seconds())
    print("Tiempo opencl: " + diff)

if __name__ == '__main__':
    basicTest(16, 16, 0.5)
