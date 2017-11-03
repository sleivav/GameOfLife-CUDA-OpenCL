import os
from datetime import datetime

from src.cuda_gol import CudaGameOfLife
from src.data_manager import DataManager
from src.opencl_gol import OpenCLGameOfLife
from src.sequential_gol import SequentialGameOfLife


def basicTest(width: int, height: int, prob: float, iterations: int):
    test_file = 'test.out'

    DataManager.generate_data(width, height, prob, test_file)

    src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
    cuda_file = os.path.join(src_path, 'cuda_gol_kernel.cu')
    opencl_file = os.path.join(src_path, 'opencl_gol_kernel.cl')

    cudaGol_small = CudaGameOfLife(test_file, program_file=str(cuda_file))
    openClGol_small = OpenCLGameOfLife(test_file, program_file=str(opencl_file))
    sequential_small = SequentialGameOfLife(test_file)

    time_start = datetime.now()
    #sequential_small.iterate(iterations)
    time_end = datetime.now()
    diff = str((time_end - time_start).total_seconds())
    print("Tiempo secuencial: " + diff)

    time_start = datetime.now()
    cudaGol_small.iterate(iterations)
    time_end = datetime.now()
    diff = str((time_end - time_start).total_seconds())
    print("Tiempo cuda: " + diff)
    #del cudaGol_small

    time_start = datetime.now()
    openClGol_small.iterate(iterations)
    time_end = datetime.now()
    diff = str((time_end - time_start).total_seconds())
    print("Tiempo opencl: " + diff)

if __name__ == "__main__":
    basicTest(2000, 2000, 0.5, 100)
