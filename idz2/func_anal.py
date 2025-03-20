from fractions import Fraction
import numpy as np
import sympy as sp


def num_to_fraction(num: str | int) -> Fraction:
    """
    Перевести число или дробь в Fraction
    """
    if isinstance(num, int):
        return Fraction(num)

    if '/' not in num:
        raise ValueError(
            'Дробь передаваемая в виде строки не содержит символа "/".' +
            f'Дробь "{num}"'
        )

    return Fraction(*map(int, num.split('/')))


def fraction_to_rational(num: Fraction) -> sp.Rational:
    """
    Перевести Fraction в Rational
    """
    return sp.Rational(num.numerator, num.denominator)


def rational_to_fraction(num: sp.Rational) -> Fraction:
    """
    Перевести Rational в Fraction
    """
    return Fraction(num.numerator, num.denominator)


def normal_matrix(matrix):
    """
    Нормализация матрицы
    """
    max_denominator = 1
    for el in (el for row in matrix for el in row):
        max_denominator = max(el.denominator, max_denominator)

    return (
        np.array([[num * max_denominator for num in row]for row in matrix]),
        max_denominator
    )


def get_matrix_fraction(matrix):
    """
    Перевод матрицы в np.array[Fraction]
    """
    return np.array([[num_to_fraction(num) for num in row] for row in matrix])


def print_fraction_matrix(matrix):
    """
    Напечатать np.array[Fraction]
    """
    sympy_matrix = sp.Matrix([
        [fraction_to_rational(num) for num in row] for row in matrix
    ])

    sp.pprint(sympy_matrix)


def find_max_eigenvalue_and_eigenvector(matrix):
    matrix = np.array([
        [float(num) for num in row] for row in matrix
    ])
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
        return norm, np.array([*map(float, vector)])

    elif p == 2:

        eigenvalue, eigenvector = \
            find_max_eigenvalue_and_eigenvector(matrix.T @ matrix)

        return np.sqrt(eigenvalue), eigenvector

    elif p == np.inf:
        index, norm = max(
            enumerate(np.sum(np.abs(matrix), axis=1)), key=lambda t: t[1]
        )
        vector = np.array([(-1 if el < 0 else 1) for el in matrix[index]])
        return norm, np.array([*map(float, vector)])


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

    # matrix = [
    #     ['333/7', '96/7', '-180/7', '12/7'],
    #     ['1431/14', '-69/7', '-477/7', '-219/14'],
    #     ['-540/7', '312/7', '423/7', '150/7'],
    #     ['-324/7', '288/7', '216/7', '153/7']
    # ]

    data = get_matrix_fraction(matrix)
    print_fraction_matrix(data)

    for p in [1, 2, np.inf]:
        norm, vector = norm_matrix(data, p)
        print(f'Норма при p = {p}: {norm}')
        print(f'Вектор при p = {p}: {vector}')
        tmp = matrix @ vector.reshape(-1, 1)
        print(f'||A * x|| = {norm_vector(tmp.flatten(), p)}')
        print()


if __name__ == '__main__':
    main()
