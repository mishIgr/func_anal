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


def find_max_eigenvalue_and_eigenvector(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    max_index = np.argmax(eigenvalues)

    max_eigenvalue = eigenvalues[max_index]
    max_eigenvector = eigenvectors[:, max_index]

    return max_eigenvalue, max_eigenvector


def norm_matrix(matrix, p):
    """
    Вычисление нормы матрицы
    """
    if p == 1:
        index, norm = max(
            enumerate(np.sum(np.abs(matrix), axis=0)), key=lambda t: t[1]
        )
        vector = np.array(
            [(ind == index) * 1 for ind in range(len(matrix[0]))]
        )
        return norm, vector

    elif p == 2:

        eigenvalue, eigenvector = \
            find_max_eigenvalue_and_eigenvector(matrix.T @ matrix)

        return np.sqrt(eigenvalue), eigenvector

    elif p == np.inf:
        index, norm = max(
            enumerate(np.sum(np.abs(matrix), axis=1)), key=lambda t: t[1]
        )
        vector = np.array([(-1 if el < 0 else 1) for el in matrix[index]])
        return norm, vector


def norm_vector(vector, p):
    """
    Вычисление нормы вектора
    """
    if p == 1:
        return np.sum(np.abs(vector))

    elif p == 2:
        return np.sqrt(np.sum(vector ** 2))

    elif p == np.inf:
        return np.max(np.abs(vector))


def main():
    matrix = [
        [645, -1422, 954, -1266],
        [-684, 1521, -1008, 1332],
        [-1074, 2412, -1611, 2154],
        [246, -522, 342, -435],
    ]

    data = np.array(matrix)
    rdata = np.linalg.inv(data)

    print('Для A')
    print_matrix(data)

    for p in [1, 2, np.inf]:
        norm, vector = norm_matrix(data, p)
        print(f'Норма при p = {p}: {norm}')
        print(f'Вектор при p = {p}: {vector}')
        tmp = data @ vector.reshape(-1, 1)
        print(f'||A * x|| = ||{tmp.flatten()}|| = {norm_vector(tmp.flatten(), p)}')
        print()

    print()

    print('Для A^-1')
    print_matrix(rdata)

    for p in [1, 2, np.inf]:
        norm, vector = norm_matrix(rdata, p)
        print(f'Норма при p = {p}: {norm}')
        print(f'Вектор при p = {p}: {vector}')
        tmp = rdata @ vector.reshape(-1, 1)
        print(f'||A * x|| = ||{tmp.flatten()}|| = {norm_vector(tmp.flatten(), p)}')
        print()

    print()

    print('Обусловленость A')
    for p in [1, 2, np.inf]:
        norm, _ = norm_matrix(rdata, p)
        rnorm, _ = norm_matrix(rdata, p)
        print(f'p = {p}: {norm * rnorm}')

    print()

    print('Союзная матрица')
    g = data.T @ data
    print_matrix(g, precision=-1)
    eigenvalues, eigenvectors = np.linalg.eig(g)
    print(f'Собственнные значения: {eigenvalues}')
    print(f'Собственнные векторы: {eigenvectors}')
    print()

    print(np.sqrt(np.max(eigenvalues)))


if __name__ == '__main__':
    main()
