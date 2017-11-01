import numpy as np


class DataManager:
    @staticmethod
    def generate_data(width: int, height: int, dead_prob: float):
        generated_matrix = np.random.choice(2, size=(width, height),
                                            p=[dead_prob, 1-dead_prob])
        np.savetxt('test.out', generated_matrix, delimiter=',')
        return generated_matrix

    @staticmethod
    def load_data(input_file: str):
        return np.loadtxt(input_file, delimiter=',')
