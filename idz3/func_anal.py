import numpy as np
from sympy import symbols, Eq, solve, Matrix, pprint, Symbol
from sympy.core.numbers import Rational
from typing import Mapping, Union, Any, Callable


def get_symbols(c: str, num: int = 4) -> tuple[Symbol, ...]:
    """
    Генерирует кортеж символьных переменных

    Args:
        c: Буква для имен переменных
        num: Количество переменных

    Returns:
        Кортеж символьных переменных (c1, c2, ..., cN)
    """
    return symbols(
        ' '.join((c + str(i + 1) for i in range(num)))
    )


def get_slar(sys_eq: tuple[list | tuple | Matrix, ...]) -> \
        Callable[[*tuple[Symbol, ...]], tuple[dict[Symbol, Any], list[Eq]]]:
    """
    Создает функцию для решения системы линейных уравнений
    """
    processed_eq = []
    for eq in sys_eq:
        if isinstance(eq, tuple):
            vector, value = eq
            processed_eq.append((Matrix(vector), value))
        else:
            processed_eq.append((Matrix(eq), 0))

    def solve_slar(*x_vars: Any) -> tuple[dict[Any, Any], list[Eq]]:
        """
        Решает систему линейных уравнений
        """
        x = Matrix(x_vars)  # Вектор x из переданных переменных

        equations = []
        for eq in processed_eq:
            equations.append(Eq(eq[0].dot(x), eq[1]))

        solution = solve(equations, x_vars)
        return solution, equations

    return solve_slar


def convert_solve_to_np(solution: dict[Any, Any],
                        values: Mapping[Symbol, Union[int, float]],
                        *x_vars: Any,
                        dtype: type = np.float32) -> np.ndarray:
    """
    Конвертирует символьное решение в numpy array с плавающей точкой
    """
    float_solution = {}

    for var in x_vars:
        if var in solution:
            val = solution[var].subs(values)

            if val.is_integer:
                float_solution[var] = float(int(val))
            elif isinstance(val, Rational):
                float_solution[var] = float(val.p) / float(val.q)
            else:
                float_solution[var] = float(val)
        else:
            float_solution[var] = float(values[var])

    return np.array([float_solution[var] for var in x_vars], dtype=dtype)


def get_norm_ort_vector(c: str,
                        *sys_eq: tuple | np.ndarray,
                        verbose: bool = False) -> np.ndarray:
    """
    Находит нормированный ортогональный вектор
    """
    x_vars = get_symbols(c)
    slar = get_slar(sys_eq)
    solution, equations = slar(*x_vars)
    values_substitution = {v: 1 for v in x_vars if v not in solution}
    x = convert_solve_to_np(solution, values_substitution, *x_vars)

    if verbose:
        print(f'Для нахождения {c} решим систему:')
        for equation in equations:
            pprint(equation)
        print('Получим решение:')
        print(solution)
        print('Подставим значения:')
        print(values_substitution)
        print('Получим вектор:')
        print(x)
        print()

    return x


def exam_vector(g: np.ndarray,
                k: np.ndarray,
                a: np.ndarray,
                b: np.ndarray,
                c: np.ndarray) -> None:
    """
    Проверяет свойства полученного базиса
    """
    print('Проверка получившегося базиса:')
    print(f'g.dot: {g.dot(a)} {g.dot(b)} {g.dot(c)}')
    print(f'k.dot: {k.dot(a)} {k.dot(b)} {k.dot(c)}')
    print(f'ортогональность: {a.dot(b)} {a.dot(c)} {b.dot(c)}')
    print(f'нормированность: {np.linalg.norm(a)} '
          f'{np.linalg.norm(b)} {np.linalg.norm(c)}')
    print()


def main():
    verbose = True
    k = np.array([1, 2, 3, 9])
    g = np.array([3, 6, 1, 2])

    g_norm = np.linalg.norm(g)
    print(f'Норма g в X: {g_norm}')
    print()

    a = get_norm_ort_vector('a', g, k, verbose=verbose)
    a /= np.linalg.norm(a)
    b = get_norm_ort_vector('b', g, k, a, verbose=verbose)
    b /= np.linalg.norm(b)
    c = get_norm_ort_vector('c', k, a, b, verbose=verbose)
    c /= np.linalg.norm(c)

    print(f'Базис: {a}, {b}, {c}')
    print(f'g(c) = {g.dot(c)}')
    print()

    exam_vector(g, k, a, b, c)

    g_norm_in_y = abs(g.dot(c))
    print(f'Норма g в Y*: {g_norm_in_y}')
    print()

    d = get_norm_ort_vector('d', a, b, c, verbose=verbose)
    d /= np.linalg.norm(d)

    f = get_norm_ort_vector('f', a, b, (c, g.dot(c)), d, verbose=verbose)

    f_norm_in_x = np.abs(f.dot(c))
    print(f'Норма f в X*: {f_norm_in_x}')
    print()

    for ind in range(4):
        print(f'f(e_{ind}) = {f[ind]}')
    f_norm = np.linalg.norm(f)
    print(f'Норма f в {{e}}: {f_norm}')


if __name__ == "__main__":
    main()
