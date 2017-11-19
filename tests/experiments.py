import os
from datetime import datetime

from src.gol.cuda_gol import CudaGameOfLife
from src.gol.opencl_gol import OpenCLGameOfLife

from src.gol.data_manager import DataManager


def basicTest(width: int, height: int, prob: float, iterations: int):
    test_file = 'test.out'

    DataManager.generate_data(width, height, prob, test_file)

    src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
    cuda_file = os.path.join(src_path, 'cuda_gol_kernel.cu')
    opencl_file = os.path.join(src_path, 'opencl_gol_kernel.cl')

    cudaGol_small = CudaGameOfLife(test_file, program_file=str(cuda_file))
    openClGol_small = OpenCLGameOfLife(test_file, program_file=str(opencl_file))

    time_start = datetime.now()
    cudaGol_small.iterate(iterations)
    time_end = datetime.now()
    diff = str((time_end - time_start).total_seconds())
    print("Tiempo cuda: " + diff)

    time_start = datetime.now()
    openClGol_small.iterate(iterations)
    time_end = datetime.now()
    diff = str((time_end - time_start).total_seconds())
    print("Tiempo opencl: " + diff)


def test_from_data():
    for i in range(11):
        for j in range(10):
            curr = 2 ** (i+1)
            currstr = str(curr)

            file = currstr + "x" + currstr
            src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
            cuda_file = os.path.join(src_path, 'cuda_gol_kernel.cu')
            opencl_file = os.path.join(src_path, 'opencl_gol_kernel.cl')

            cudaGol_small = CudaGameOfLife(file, program_file=str(cuda_file))
            openClGol_small = OpenCLGameOfLife(file, program_file=str(opencl_file))

            iterations = 1000

            out_cuda = open("out_cuda.csv", "a")
            out_opencl = open("out_opencl.csv", "a")

            time_start = datetime.now()
            cudaGol_small.iterate(iterations)
            time_end = datetime.now()
            diff = str((time_end - time_start).total_seconds())
            out_cuda.write(currstr + ", " + diff + "\n")
            print("n:" + currstr + "\nTiempo cuda: " + diff)

            time_start = datetime.now()
            openClGol_small.iterate(iterations)
            time_end = datetime.now()
            diff = str((time_end - time_start).total_seconds())
            out_opencl.write(currstr + ", " + diff + "\n")
            print("Tiempo opencl: " + diff + "\niteraciones:" + currstr)


if __name__ == "__main__":
    test_from_data()
