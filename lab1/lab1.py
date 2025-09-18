import numpy as np

SEED = 123


def generate_uniform_distribution(low: int, high: int, seed: int | None = None) -> np.ndarray:
    """
    Функция для генерации выборки из равномерного закона распределения на интервале от [0, high)

    Args:
        low: нижняя граница (TODO сделать возможность прописывать нижнюю границу)
        high: верхняя граница
        seed: сид генерации случайных чисел

    """
    if seed is None:
        ran_gen = np.random.default_rng()
    else:
        ran_gen = np.random.default_rng(seed)

    return ran_gen.random(high)

s = generate_uniform_distribution(0, 1, seed=SEED)
print("min>=0?", float(s.min()) >= 0.0)
print("max<=1?", float(s.max()) <= 1.0)
print("пример:", s[:5])