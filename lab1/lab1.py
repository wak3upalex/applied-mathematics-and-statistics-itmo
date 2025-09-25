from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# seed number can be set to any custom one (just to recreate results of work
SEED = 123


def generate_uniform_distribution(n: int, seed: int | None = None) -> np.ndarray:
    """
    Функция для генерации выборки из равномерного закона распределения на интервале от [0, 1)

    Args:
        n: кол-во чисел для генерации
        seed: сид генерации случайных чисел
    Returns:
        np.ndarray: массив случайных чисел
    """
    if seed is None:
        ran_gen = np.random.default_rng()
    else:
        ran_gen = np.random.default_rng(seed)

    return ran_gen.random(n)


def ecdf_points(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Функция для нахождения точек/ступеней для выборочной функции распределения
    Args:
        sample: выборка значений

    Returns:
        tuple[np.ndarray, np.ndarray]: отсортированный список выборки значений для функции выборочной функции распределения, список значений самой функции распределения по значениям выборки
    """
    x_sorted = np.sort(sample)  # сортируем выборку

    n = len(x_sorted)
    # Ступеньки ЭФР: массив [1/n, 2/n, …, n/n].
    # Это значения F_n(x(i)).

    y = np.arange(1, n + 1) / n
    return x_sorted, y


def plot_ecdf_vs_theory(sample: np.ndarray, title: str = "") -> None:
    """
    Функция для рисования ЭФР и теоритической функции F(x) = x на одном графике для наглядности и сравнения
    Args:
        sample: выборка
        title: необязательный заголовок

    Returns:
        None: выводит график
    """
    # получаем точки для ЭФР
    x_sort_plot, y = ecdf_points(sample)
    x_plot = np.r_[0.0, x_sort_plot, 1.0]
    y_plot = np.r_[0.0, y, 1.0]
    # генеральная F(x)=x на [0,1]
    grid = np.linspace(0, 1, 500)

    plt.figure()
    # Рисуем ступенчатую функцию: where="post" делает её правонепрерывной (скачок происходит после точки x_(i)) — так по определению устроена ЭФР.
    plt.step(x_plot, y_plot, where="post", label="Эмпирическая Fₙ")
    # генеральная функция распределения
    plt.plot(grid, grid, label="генеральная F(x)=x")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.title(title or f"ECDF и генеральная функция распределения (n={len(sample)})")
    plt.legend()
    plt.show()


for n in [10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4]:
    # генерируем выборку
    s = generate_uniform_distribution(n, seed=SEED)
    print(len(s))
    print("min>=0?", float(s.min()) >= 0.0)
    print("max<=1?", float(s.max()) <= 1.0)
    print("пример:", s[:5])

    # Считаем точки для ЭФР по данной выборки
    x_sort, Fn_x = ecdf_points(s)
    print("Неубывающая", (np.all(np.diff(x_sort) >= 0)))
    plot_ecdf_vs_theory(s, f"ECDF vs F(x)=x, n={n}")
