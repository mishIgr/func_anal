import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


INDICES_OF_PLANES = [
    [1, 2, 5],
    [2, 3, 5],
    [3, 4, 5],
    [4, 1, 5],
    [1, 2, 6],
    [2, 3, 6],
    [3, 4, 6],
    [4, 1, 6],
]

CONE_OF_VERTICES = [
    [1, 4, 5, 8],
    [1, 2, 5, 6],
    [2, 3, 6, 7],
    [3, 4, 7, 8],
    [1, 2, 3, 4],
    [5, 6, 7, 8],
]

ZERO_POINT = np.array(
    [0 for _ in range(3)]
)


class Plane:
    """
    Класс, реализующий плоскость
    """

    index_d = 3

    def __init__(self, points: list[np.ndarray]):

        if len(points) != 3:
            raise ValueError('Для построения плоскости нужно ровно 3 точки')

        self.data = np.zeros(4)

        normal = np.cross(points[1] - points[0], points[2] - points[0])
        self.data[:self.index_d] = normal

        self.data[self.index_d] = -np.dot(normal, points[0])

    def substitute_point(self, point: np.ndarray) -> float:
        return np.dot(
            point,
            self.data[:self.index_d]
        ) + self.data[self.index_d]

    def orientation(self, point: np.ndarray | list):
        if isinstance(point, list):
            point = np.array(point)

        if self.substitute_point(point) > 0:
            self.data[:self.index_d] *= -1

    @property
    def normal(self) -> np.ndarray:
        return self.data[:self.index_d]

    def __str__(self) -> str:
        data_str = map(str, self.data)
        merge_data_str_for_var = zip(data_str, ['x', 'y', 'z', ''])
        monomials_list = map(
            lambda t: t[0] + ' * ' + t[1] if t[1] else t[0],
            merge_data_str_for_var
        )
        plane_str = ' + '.join(monomials_list)
        return plane_str


def another_index(index: list[int]):
    max_index = max(index) + 1
    for ind in range(1, max_index):
        if ind not in index:
            return ind
    return max_index


def get_orientation_normals(data: np.ndarray) -> list[np.ndarray]:
    normals = []

    for indexes in INDICES_OF_PLANES:
        points = [data[ind - 1] for ind in indexes]
        plane = Plane(points)
        plane.orientation(data[another_index(indexes) - 1])
        normals.append(plane.normal)

    return normals


class Coord(Enum):
    """
    Координаты в векторе
    """
    X = 0
    Y = 1
    Z = 2


def coef_coord_basis_expansion(
        basis: list[np.ndarray],
        vector: np.ndarray,
        ind: int
) -> float:
    """
    Определяет коэффициент в разложении по базису под номером ind
    """
    other_coord = filter(lambda c: c != ind, range(3))
    orthogonal = np.cross(basis[next(other_coord)], basis[next(other_coord)])
    coef = np.dot(orthogonal, vector) / np.dot(orthogonal, basis[ind])
    return coef


def basis_expansion(
        basis: list[np.ndarray], vector: np.ndarray
) -> list[float]:
    """
    Определяет коэффициенты в разложении vector в базисе basis
    """
    coefficients = [
        coef_coord_basis_expansion(basis, vector, ind)
        for ind in range(3)
    ]

    return coefficients


def divide_two_dividing_point(cone: list[int], normals: list[np.ndarray]) -> \
        list[int]:
    first_ind = cone[0]
    first_point = np.array(normals[first_ind - 1])
    zero_point = ZERO_POINT

    for ind in range(1, len(cone)):
        point = np.array(normals[cone[ind] - 1])
        plane = Plane([first_point, point, zero_point])
        tmp = 1.
        for i in cone:
            if i not in [first_ind, cone[ind]]:
                tmp *= plane.substitute_point(normals[i - 1])
        if tmp < 0:
            return [first_ind, cone[ind]]

    return [-1, -1]


def divide_into_two_planes(cone: list[int], normals: list[np.ndarray]) -> \
        tuple[list[int], list[int]]:
    dividing_points = divide_two_dividing_point(cone, normals)

    first_point, second_point = filter(
        lambda ind: ind not in dividing_points, cone
    )
    return dividing_points + [first_point], dividing_points + [second_point]


def find_extremes(func: np.ndarray, data: np.ndarray,
                  normals: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    max_extremum = np.zeros(3)
    min_extremum = np.zeros(3)

    for ind, cone in enumerate(CONE_OF_VERTICES):
        first_plane, second_plane = divide_into_two_planes(cone, normals)

        for plane in [first_plane, second_plane]:
            basis_vector = [*map(lambda ind: normals[ind - 1], plane)]
            expansion = basis_expansion(basis_vector, func)
            if all(map(lambda x: x >= 0, expansion)):
                max_extremum = data[ind]
            if all(map(lambda x: x <= 0, expansion)):
                min_extremum = data[ind]

    return min_extremum, max_extremum


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
    data = np.array([
        [11, 8, 6],
        [-1, 14, 6],
        [-1, 6, 6],
        [11, 0, 6],
        [5, 7, 11],
        [5, 7, 3],
    ])

    h = np.array([26, 16, 54])

    # draw_fig(data)

    normals = get_orientation_normals(data)
    min_extremum, max_extremum = find_extremes(h, data, normals)
    print(f'{min_extremum=} {max_extremum=}')


if __name__ == '__main__':
    main()
