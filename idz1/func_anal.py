import numpy as np
from enum import Enum
from fractions import Fraction
from tabulate import tabulate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


INDICES_OF_PLANES = [
    (1, 2, 3),
    (1, 2, 4),
    (1, 3, 5),
    (2, 3, 6),
    (3, 5, 7),
    (3, 7, 8),
    (3, 6, 8),
    (7, 8, 9),
    (11, 12, 16),
    (8, 11, 16),
    (8, 9, 16),
    (6, 8, 11),
    (10, 11, 12),
    (2, 10, 11),
    (2, 6, 11),
    (2, 4, 10),
    (1, 5, 14),
    (1, 13, 14),
    (13, 14, 15),
    (1, 4, 13),
    (7, 14, 17),
    (14, 15, 17),
    (5, 7, 14),
    (7, 9, 17),
    (9, 16, 17),
    (16, 17, 18),
    (12, 16, 18),
    (15, 17, 18),
    (10, 12, 18),
    (13, 15, 18),
    (10, 13, 18),
    (4, 10, 13)
]

ZERO_POINT = np.array(
    [Fraction(0) for _ in range(3)]
)


class Plane:
    """
    Класс, реализующий плоскость
    """

    class Coefficient(Enum):
        """
        Индексы коэффициентов в списке
        """
        A = 0
        B = 1
        C = 2
        D = 3

    def __init__(self, points: list[np.ndarray]):

        if len(points) != 3:
            raise ValueError('Для построения плоскости нужно ровно 3 точки')

        self.data = [Fraction(0) for _ in range(4)]

        normal = np.cross(points[1] - points[0], points[2] - points[0])
        self.data[Plane.Coefficient.A.value:Plane.Coefficient.D.value] = normal

        self.data[Plane.Coefficient.D.value] = -np.dot(normal, points[0])

        if self.substitute_point(ZERO_POINT) > 0:
            self.data = [num * -1 for num in self.data]

    def substitute_point(self, point: np.ndarray) -> Fraction:
        return np.dot(
            point,
            np.array(
                self.data[Plane.Coefficient.A.value:Plane.Coefficient.D.value]
            )
        ) + self.data[Plane.Coefficient.D.value]

    def __str__(self) -> str:
        data_str = map(str, self.data)
        merge_data_str_for_var = zip(data_str, ['x', 'y', 'z', ''])
        monomials_list = map(
            lambda t: t[0] + ' * ' + t[1] if t[1] else t[0],
            merge_data_str_for_var
        )
        plane_str = ' + '.join(monomials_list)
        return plane_str


class Coord(Enum):
    """
    Координаты в векторе
    """
    X = 0
    Y = 1
    Z = 2


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


def convert_to_numpy_data(data: list[list[str | int]]) -> list[np.ndarray]:
    """
    Переводит массив с точками в массив с векторами, хранящими Fraction
    """
    np_data = [np.array([*map(num_to_fraction, vector)]) for vector in data]
    return np_data


def change_sign(vector: np.ndarray, coord: Coord) -> np.ndarray:
    """
    Меняет знак числа coord в vector
    """
    change_sign_arr = np.array([1] * 3)
    change_sign_arr[coord.value] = -1
    return vector * change_sign_arr


def extend_negative_coord_points(
        write_data: list[np.ndarray],
        read_data: list[np.ndarray],
        coord: Coord
) -> list[np.ndarray]:
    """
    Добавляет точки, которое соответствуют условию меняя знак coord:
    read_data[coord] > 0
    """

    extend_data = write_data[:]

    extend_data.extend(
        map(
            lambda v: change_sign(v, coord),
            filter(lambda v: v[coord.value], read_data)
        )
    )

    return extend_data


def add_new_point(data: list[np.ndarray]) -> list[np.ndarray]:
    """
    Дополняет массив точками (+-x, +-y, +-z)
    """
    new_data = data[:]

    for coord in [Coord.X, Coord.Y, Coord.Z]:
        new_data = extend_negative_coord_points(new_data, data, coord)

    new_data.extend(map(lambda v: v * -1, data[:3]))

    return new_data


def build_planes(data: list[np.ndarray]) -> list[Plane]:
    """
    По точкам строит плоскости
    """
    planes = []

    for indexes in INDICES_OF_PLANES:
        plane = Plane([*map(lambda ind: data[ind - 1], indexes)])
        planes.append(plane)

    return planes


def print_plane_point_table(
        planes: list[Plane], convex_np_data: list[np.ndarray]
):
    """
    Печатает таблицу с использованием tabulate, где строки -
    это номера плоскостей,
    а столбцы - номера точек. Пересечение строки и столбца -
    это результат метода substitute_point.
    """
    headers = [f'Po{i + 1}' for i in range(len(convex_np_data))]

    table = []
    for ind, plane in enumerate(planes):
        row = [f'{plane.substitute_point(vector)}' for vector in convex_np_data]
        table.append([f'Pl{ind + 1}'] + row)

    print(tabulate(table, headers=["Plane"] + headers, tablefmt="grid"))


def coef_coord_basis_expansion(
        basis: list[np.ndarray],
        vector: np.ndarray,
        ind: int
) -> Fraction:
    """
    Определяет коэффициент в разложении по базису под номером ind
    """
    other_coord = filter(lambda c: c != ind, range(3))
    orthogonal = np.cross(basis[next(other_coord)], basis[next(other_coord)])
    coef = np.dot(orthogonal, vector) / np.dot(orthogonal, basis[ind])
    return coef


def basis_expansion(
        basis: list[np.ndarray], vector: np.ndarray
) -> list[Fraction]:
    """
    Определяет коэффициенты в разложении vector в базисе basis
    """
    coefficients = [
        coef_coord_basis_expansion(basis, vector, ind)
        for ind in range(3)
    ]

    return coefficients


def norm_w(data: list[np.ndarray], vector: np.ndarray) -> Fraction:
    """
    Определяет норму Минковского определённую многогранником data
    """
    for indexes in INDICES_OF_PLANES:
        basis = [*map(lambda ind: data[ind - 1], indexes)]
        coefs = basis_expansion(basis, vector)

        if all(map(lambda num: num >= 0, coefs)):
            print(
                f'Конус граней образован точками под номерами {indexes}'
                f'А точнее {[*map(lambda ind: data[ind - 1], indexes)]}\n'
                f'Вектор {[*map(str, vector)]} имеет разложение с '
                f'коэффициентами: {", ".join(map(str, coefs))}'
            )
            return sum(coefs)

    return Fraction(-1)


def draw_fig(points):
    """Создание 3D-графика"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Построение каждой плоскости
    for plane in INDICES_OF_PLANES:
        # Получаем координаты точек плоскости
        triangle = [points[i - 1] for i in plane]
        # Создаём полигон
        poly = Poly3DCollection([triangle], alpha=0.9, edgecolor='k')
        ax.add_collection3d(poly)

    # Подписываем номера точек
    for i, point in enumerate(points):
        ax.text(point[0], point[1], point[2], f'{i + 1}', color='red',
                fontsize=12)

    # Установка пределов осей
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])
    ax.set_zlim([-20, 20])

    # Подписи осей
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def main():
    data = [
        [9, 8, 0],
        [4, 0, 9],
        [0, 8, 2],
        ["107 / 4", 0, 0],
        [0, "640 / 71", 0],
        [0, 0, "85 / 9"],
    ]

    a = [8, -7, 8]
    b = [-7, 5, 5]

    np_data = convert_to_numpy_data(data)
    convex_np_data = add_new_point(np_data)

    # draw_fig(convex_np_data)

    for ind, vector in enumerate(convex_np_data):
        print(f'point {ind + 1}: {" ".join(map(str, vector))}')

    planes = build_planes(convex_np_data)
    for ind, plane in enumerate(planes):
        print(f'plane {ind + 1}: {plane}')

    print_plane_point_table(planes, convex_np_data)

    a_fraction_positive = [*map(lambda num: num_to_fraction(abs(num)), a)]
    b_fraction_positive = [*map(lambda num: num_to_fraction(abs(num)), b)]

    sum_a_b = [a[ind] + b[ind] for ind in range(3)]
    ab_fraction_positive = [
        *map(lambda num: num_to_fraction(abs(num)), sum_a_b)
    ]

    for vector in [a_fraction_positive, b_fraction_positive, ab_fraction_positive]:
        print(f'Для {vector}')
        for indexes in INDICES_OF_PLANES[:4]:
            basis = [*map(lambda ind: np_data[ind - 1], indexes)]
            coefs = basis_expansion(basis, vector)
            print(f'Разложение в {indexes}: {coefs}')
        print()

    print()
    norm_a = norm_w(np_data, a_fraction_positive)
    print()
    norm_b = norm_w(np_data, b_fraction_positive)
    print()
    norm_ab = norm_w(np_data, ab_fraction_positive)
    print()

    print(f'Норма {a} равна: {norm_a}')
    print(f'Норма {b} равна: {norm_b}')
    print(f'Норма {sum_a_b} равна: {norm_ab}')

    print(f'Проверка неравенства: {norm_ab <= norm_a + norm_b}')


if __name__ == '__main__':
    main()
