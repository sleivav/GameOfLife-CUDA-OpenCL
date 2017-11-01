import numpy as np


def generate_data(width: int, height: int, dead_prob: float):
    generated_matrix = np.random.choice(2, size=(width, height),
                                        p=[dead_prob, 1-dead_prob])
    np.savetxt('test.out', generated_matrix, delimiter=',')
    return generated_matrix


def load_data():
    return np.loadtxt('test.out', delimiter=',')
