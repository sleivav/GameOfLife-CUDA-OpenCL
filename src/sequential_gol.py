from src.gol import GameOfLife


class SequentialGameOfLife(GameOfLife):
    def __init__(self, input_file: str, program_file: str = None):
        super().__init__(input_file, program_file)

    def compute(self, matrix_width: int, matrix_height: int):
        prev = self.data_matrix.copy()
        for y in range(matrix_height):
            y0 = (y + matrix_height - 1) % matrix_height
            y1 = (y + 1) % matrix_height
            for x in range(matrix_height):
                x0 = (x + matrix_width - 1) % matrix_width
                x1 = (x + 1) % matrix_width
                alive_cells = self.count_alive_cells(x0, x, x1,
                                                     y0, y, y1, prev)
                if alive_cells == 3 or(alive_cells == 2 and (self.data_matrix.item((x, y)) == 1)):
                    self.result_matrix[x, y] = 1
                else:
                    self.result_matrix[x, y] = 0
        self.data_matrix = self.result_matrix

    @staticmethod
    def count_alive_cells(x0: int, x: int, x1: int, y0: int, y: int, y1: int,
                          matrix):
        return matrix.item(x0, y0) + matrix.item(x, y0) + matrix.item(x1, y0) +\
               matrix.item(x0, y) + matrix.item(x1, y) +\
               matrix.item(x0, y1) + matrix.item(x, y1) + matrix.item(x1, y1)

    def iterate(self, iterations: int):
        for x in range(iterations):
            self.compute(self.width, self.height)
