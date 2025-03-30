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
    G = np.array([
        [2097873.0, -4676454.0, 3119148.0, -4148064.0],
        [-4676454.0, 10425753.0, -6954012.0, 9248742.0],
        [3119148.0, -6954012.0, 4638465.0, -6169284.0],
        [-4148064.0, 9248742.0, -6169284.0, 8205921.0]
    ])

    x = np.array([1, 1, 1, 1]).reshape(-1, 1)
    b = np.array([1/2, 1/3, 1/4, 1/5]).reshape(-1, 1)

    B = np.linalg.inv(G)

    print('B:')
    print_matrix(B)
    print()

    res = []
    index_save_iter = [5, 10]

    for i in range(100):
        x = B @ x + b
        if i + 1 in index_save_iter:
            res.append(x)

    print('x: ')
    print_matrix(x)
    print()

    print('Проверка', (np.identity(4) - B) @ x - b)
    print()

    for ind, v in zip(index_save_iter, res):
        print(f'x{ind}:')
        print_matrix(v)
        print()


if __name__ == '__main__':
    main()
