from abc import ABC, abstractmethod

import numpy as np

from src.gol.data_manager import DataManager


class GameOfLife(ABC):
    @abstractmethod
    def __init__(self, input_file: str, program_file=None, block_size=0):
        if program_file is not None:
            try:
                file = open(program_file, 'r')
                self.program_file = file
            except IOError:
                raise IOError("Can't open file: " + program_file)
        self.data_matrix = DataManager.load_data(input_file)
        self.result_matrix = np.empty_like(self.data_matrix)
        self.width = self.data_matrix.shape[0]
        self.height = self.data_matrix.shape[1]
        self.block_size = block_size

    @abstractmethod
    def iterate(self, iterations: int):
        raise NotImplementedError()

