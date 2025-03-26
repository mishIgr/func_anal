import numpy as np
import sympy as sp


def print_matrix(matrix, precision=4):
    """
    Напечатать np.array[float]
    """
    sympy_matrix = sp.Matrix(matrix)

    if precision >= 0:
        sympy_matrix = sp.N(sympy_matrix, precision)

    sp.pprint(sympy_matrix)


def main():
    A = np.array([
        [645, -1422, 954, -1266],
        [-684, 1521, -1008, 1332],
        [-1074, 2412, -1611, 2154],
        [246, -522, 342, -435],
    ])

    x = np.array([1, 1, 1, 1]).reshape(-1, 1)
    b = np.array([1/2, 1/3, 1/4, 1/5]).reshape(-1, 1)

    B = np.linalg.inv(A.T @ A)

    print('B:')
    print_matrix(B)
    print()

    res = []
    index_save_iter = {5, 10}

    for i in range(100):
        x = B @ x + b
        if i + 1 in index_save_iter:
            res.append(x)

    print('x: ')
    print_matrix(x)
    print()

    for v in res:
        print_matrix(v)


if __name__ == '__main__':
    main()
