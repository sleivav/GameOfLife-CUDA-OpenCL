from abc import ABC, abstractmethod

from data_manager import DataManager
import numpy as np


class GameOfLife(ABC):
    @abstractmethod
    def __init__(self, input_file: str, program_file: str):
        self.data_matrix = DataManager.load_data(input_file)
        self.result_matrix = np.empty_like(self.data_matrix)
        self.program_file = program_file
        self.width = self.data_matrix.shape[0]
        self.height = self.data_matrix.shape[1]

    @abstractmethod
    def iterate(self, iterations: int):
        raise NotImplementedError()
