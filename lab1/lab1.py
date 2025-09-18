import numpy as np

SEED = 123


def generate_uniform_distribution(n:int, seed: int | None = None) -> np.ndarray:
    """
    Функция для генерации выборки из равномерного закона распределения на интервале от [0, high)

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





s = generate_uniform_distribution(10, seed=SEED)
print(len(s))
print("min>=0?", float(s.min()) >= 0.0)
print("max<=1?", float(s.max()) <= 1.0)
print("пример:", s[:5])