import numpy as np

from data_manager import DataManager

width = 1024
height = 1024

data_matrix = DataManager.generate_data(width, height, 0.5)
result_matrix = np.empty(shape=(width, height))


def compute(matrix_width: int, matrix_height: int):
    global data_matrix
    global result_matrix
    prev = data_matrix.copy()
    for y in range(matrix_height):
        y0 = (y + matrix_height - 1) % matrix_height
        y1 = (y + 1) % matrix_height
        for x in range(matrix_height):
            x0 = (x + matrix_width - 1) % matrix_width
            x1 = (x + 1) % matrix_width
            alive_cells = count_alive_cells(x0, x, x1, y0, y, y1, prev)
            if alive_cells == 3 or (alive_cells == 2 and (data_matrix.item((x, y)) == 1)):
                result_matrix[x, y] = 1
            else:
                result_matrix[x, y] = 0
    data_matrix = result_matrix


def count_alive_cells(x0: int, x: int, x1: int, y0: int, y: int, y1: int, matrix):
    return matrix.item(x0, y0) + matrix.item(x, y0) + matrix.item(x1, y0) +\
           matrix.item(x0, y) + matrix.item(x1, y) +\
           matrix.item(x0, y1) + matrix.item(x, y1) + matrix.item(x1, y1)


def iterate(iterations: int):
    global data_matrix
    global result_matrix
    for x in range(iterations):
        compute(width, height)
    return 0
