# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Лабораторная работа №2: Ядерные оценки плотности распределения
#
# Ниже решение сделано в максимально простой логике.
# Я специально не использую сложные приёмы, чтобы было видно, как каждый пункт задания
# превращается в понятные шаги в коде.
#
# Как будем понимать задачу:
# - **гистограмма** показывает, сколько наблюдений попало в каждый интервал;
# - **KDE** (ядерная оценка плотности) строит более гладкую кривую плотности;
# - **квантиль уровня `q`** это такое число, левее которого находится примерно доля `q`
#   всех наблюдений;
# - **дисперсия оценки** показывает, насколько сильно эта оценка “скачет” от выборки к
#   выборке. Чем дисперсия меньше, тем метод стабильнее.
#
# В пункте 4 мы будем делать очень прямую процедуру:
# 1. Сгенерировали выборку.
# 2. Посчитали квантили тремя способами.
# 3. Повторили это `1000` раз.
# 4. Для каждого способа посмотрели, насколько менялись полученные оценки.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, stats

SEED = 42
SAMPLE_SIZES = [10, 100, 1000, 10000]
Q_LEVELS = np.array([0.01, 0.05, 0.50], dtype=float)
Q_LABELS = {0.01: "1%", 0.05: "5%", 0.50: "50%"}
N_REPEATS = 1000

DISTRIBUTIONS = {
    "normal": lambda n, rng: rng.normal(loc=0.0, scale=1.0, size=n),
    "uniform": lambda n, rng: rng.uniform(low=0.0, high=1.0, size=n),
}
DIST_LABELS = {
    "normal": "N(0,1)",
    "uniform": "U[0,1]",
}

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25


# %%
def generate_sample(dist_name, n, rng):
    return DISTRIBUTIONS[dist_name](n, rng)


def hist_bin_count(n):
    return max(1, int(round(1 + 1.59 * np.log(n))))


def ecdf_quantiles(x, q_levels=Q_LEVELS):
    """
    Квантиль по выборочной функции распределения.

    Делаем буквально то, что следует из определения:
    - сортируем выборку;
    - ищем первый индекс, на котором накопленная доля достигает q;
    - берём соответствующий элемент выборки.
    """
    sorted_x = np.sort(np.asarray(x, dtype=float))
    n = len(sorted_x)
    indices = np.ceil(q_levels * n).astype(int) - 1
    indices = np.clip(indices, 0, n - 1)
    return sorted_x[indices]


def hist_quantiles(x, q_levels, k):
    """
    Квантиль по гистограмме.

    Идея простая:
    - строим гистограмму;
    - считаем накопленные доли по столбцам;
    - находим столбец, где впервые достигается нужный q;
    - внутри этого столбца считаем распределение равномерным.
    """
    counts, edges = np.histogram(x, bins=k)
    probs = counts / counts.sum()
    cumulative = np.cumsum(probs)
    quantiles = []

    for q in q_levels:
        bin_index = int(np.searchsorted(cumulative, q, side="left"))
        bin_index = min(bin_index, len(counts) - 1)

        prev_cdf = cumulative[bin_index - 1] if bin_index > 0 else 0.0
        bin_prob = probs[bin_index]

        if bin_prob == 0:
            quantiles.append(edges[bin_index])
            continue

        share_inside_bin = (q - prev_cdf) / bin_prob
        left_edge = edges[bin_index]
        right_edge = edges[bin_index + 1]
        quantiles.append(left_edge + share_inside_bin * (right_edge - left_edge))

    return np.asarray(quantiles, dtype=float)


def kde_grid_and_pdf(x, grid_size=800):
    """
    Строим KDE с гауссовским ядром и возвращаем:
    - сетку по x;
    - значения плотности на этой сетке.
    """
    x = np.asarray(x, dtype=float)
    std = float(np.std(x, ddof=1))

    if not np.isfinite(std) or std == 0.0:
        center = float(x[0])
        grid = np.linspace(center - 1.0, center + 1.0, grid_size)
        pdf = np.zeros_like(grid)
        pdf[len(pdf) // 2] = 1.0
        return grid, pdf

    kde = stats.gaussian_kde(x, bw_method="scott")
    left = float(np.min(x) - 4.0 * std)
    right = float(np.max(x) + 4.0 * std)
    grid = np.linspace(left, right, grid_size)
    pdf = kde(grid)
    return grid, pdf


def kde_quantiles(x, q_levels=Q_LEVELS, grid_size=800):
    """
    Квантиль по KDE.

    Здесь мы делаем три простых шага:
    - строим гладкую плотность;
    - численно интегрируем её слева направо, получая CDF;
    - по этой CDF находим нужные квантили.
    """
    grid, pdf = kde_grid_and_pdf(x, grid_size=grid_size)
    cdf = integrate.cumulative_trapezoid(pdf, grid, initial=0.0)

    if cdf[-1] <= 0:
        return np.full_like(q_levels, np.nan, dtype=float)

    cdf = cdf / cdf[-1]
    return np.interp(q_levels, cdf, grid)


def print_table(rows, headers):
    widths = [len(h) for h in headers]

    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))

    fmt = " | ".join("{:" + str(width) + "}" for width in widths)
    sep = "-+-".join("-" * width for width in widths)

    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))


def run_monte_carlo_for_one_case(dist_name, n, repeats, rng):
    """
    Повторяет один и тот же эксперимент много раз для одной пары
    (распределение, размер выборки).

    Возвращает словарь:
    - ключ это метод;
    - значение это массив размера [repeats, число_квантилей].
    """
    k = hist_bin_count(n)
    estimates = {
        "ECDF": np.empty((repeats, len(Q_LEVELS)), dtype=float),
        "Histogram": np.empty((repeats, len(Q_LEVELS)), dtype=float),
        "KDE": np.empty((repeats, len(Q_LEVELS)), dtype=float),
    }

    for repeat_index in range(repeats):
        x = generate_sample(dist_name, n, rng)
        estimates["ECDF"][repeat_index] = ecdf_quantiles(x, Q_LEVELS)
        estimates["Histogram"][repeat_index] = hist_quantiles(x, Q_LEVELS, k)
        estimates["KDE"][repeat_index] = kde_quantiles(x, Q_LEVELS)

    return estimates


# %% [markdown]
# ## Пункт 1. Генерация выборок из `N(0,1)` и `U[0,1]`
#
# Здесь всё буквально:
# - для нормального распределения используем `rng.normal(0, 1, size=n)`;
# - для равномерного распределения используем `rng.uniform(0, 1, size=n)`.
#
# В таблице ниже я вывожу для примера выборочное среднее и выборочную дисперсию.
# Они не обязаны быть ровно равны теоретическим значениям, потому что выборка случайна.
# Но при росте `n` они обычно становятся ближе к ожидаемым значениям.

# %%
demo_rng = np.random.default_rng(SEED)
demo_samples = {}
sample_rows = []

for dist_name in ("normal", "uniform"):
    for n in SAMPLE_SIZES:
        sample = generate_sample(dist_name, n, demo_rng)
        demo_samples[(dist_name, n)] = sample
        sample_rows.append(
            (
                DIST_LABELS[dist_name],
                n,
                f"{np.mean(sample):.4f}",
                f"{np.var(sample, ddof=1):.4f}",
            )
        )

print_table(
    sample_rows,
    headers=["Распределение", "n", "Выборочное среднее", "Выборочная дисперсия"],
)


# %% [markdown]
# ## Пункт 2. Гистограмма и KDE
#
# Для каждого распределения и каждого размера выборки:
# - строим гистограмму;
# - число столбцов берём по формуле `k ≈ 1 + 1.59 ln(n)`;
# - поверх рисуем KDE с гауссовским ядром.
#
# Что важно понимать:
# - гистограмма это “ломаное” представление данных по интервалам;
# - KDE это более гладкая оценка плотности.

# %%
fig, axes = plt.subplots(2, 4, figsize=(22, 9), constrained_layout=True)

for row_index, dist_name in enumerate(("normal", "uniform")):
    for col_index, n in enumerate(SAMPLE_SIZES):
        ax = axes[row_index, col_index]
        sample = demo_samples[(dist_name, n)]
        k = hist_bin_count(n)
        grid, pdf = kde_grid_and_pdf(sample)

        ax.hist(
            sample,
            bins=k,
            density=True,
            color="#84a59d",
            edgecolor="black",
            alpha=0.55,
            label=f"Гистограмма, k={k}",
        )
        ax.plot(grid, pdf, color="#bc4749", linewidth=2.0, label="KDE")

        ax.set_title(f"{DIST_LABELS[dist_name]}, n={n}")
        if dist_name == "uniform":
            ax.set_xlim(-0.25, 1.25)

        if row_index == 0 and col_index == 0:
            ax.legend(loc="upper right")

plt.suptitle("Пункт 2: гистограмма и KDE", y=1.02)
plt.show()


# %% [markdown]
# ## Пункт 3. Квантили `1%`, `5%`, `50%` тремя способами
#
# Напомню смысл:
# - квантиль `1%` это значение, левее которого примерно `1%` наблюдений;
# - квантиль `5%` это значение, левее которого примерно `5%` наблюдений;
# - квантиль `50%` это медиана.
#
# Как именно считаются квантили здесь:
# - **ECDF**: сортируем выборку и берём первый элемент, на котором накопленная доля дошла до нужного уровня;
# - **Histogram**: ищем нужный столбец гистограммы и считаем, что внутри него масса распределена равномерно;
# - **KDE**: интегрируем гладкую оценку плотности и обращаем полученную CDF.

# %%
quantile_rows = []

for dist_name in ("normal", "uniform"):
    for n in SAMPLE_SIZES:
        sample = demo_samples[(dist_name, n)]
        k = hist_bin_count(n)

        ecdf_values = ecdf_quantiles(sample, Q_LEVELS)
        hist_values = hist_quantiles(sample, Q_LEVELS, k)
        kde_values = kde_quantiles(sample, Q_LEVELS)

        for q_index, q_level in enumerate(Q_LEVELS):
            quantile_rows.append(
                (
                    DIST_LABELS[dist_name],
                    n,
                    Q_LABELS[float(q_level)],
                    f"{ecdf_values[q_index]:.5f}",
                    f"{hist_values[q_index]:.5f}",
                    f"{kde_values[q_index]:.5f}",
                )
            )

print_table(
    quantile_rows,
    headers=["Распределение", "n", "Квантиль", "ECDF", "Histogram", "KDE"],
)


# %% [markdown]
# ## Пункт 4. Повторяем эксперимент `N = 1000` раз и считаем дисперсии
#
# Это ключевой пункт задания.
#
# Почему одного запуска недостаточно:
# - если мы один раз сгенерировали выборку, то получили только **одну** оценку квантили;
# - по одной оценке нельзя понять, насколько метод устойчив;
# - поэтому нужно много раз повторить весь эксперимент и посмотреть, как сильно оценки меняются.
#
# Что именно делаем:
# 1. Для фиксированных распределения и `n` генерируем новую выборку.
# 2. Считаем `1%`, `5%`, `50%` квантили тремя способами.
# 3. Повторяем это `1000` раз.
# 4. Для каждого способа и каждой квантили считаем выборочную дисперсию.
#
# Как интерпретировать результат:
# - маленькая дисперсия означает, что метод даёт более стабильные оценки;
# - большая дисперсия означает, что оценки сильнее зависят от конкретной случайной выборки.

# %%
mc_rng = np.random.default_rng(SEED + 2026)
variance_rows = []
best_method_counts = {"ECDF": 0, "Histogram": 0, "KDE": 0}

for dist_name in ("normal", "uniform"):
    for n in SAMPLE_SIZES:
        estimates = run_monte_carlo_for_one_case(
            dist_name=dist_name,
            n=n,
            repeats=N_REPEATS,
            rng=mc_rng,
        )

        variances = {
            method_name: np.var(method_estimates, axis=0, ddof=1)
            for method_name, method_estimates in estimates.items()
        }

        for q_index, q_level in enumerate(Q_LEVELS):
            method_vars = {
                method_name: float(variances[method_name][q_index])
                for method_name in ("ECDF", "Histogram", "KDE")
            }
            best_method = min(method_vars, key=method_vars.get)
            best_method_counts[best_method] += 1

            variance_rows.append(
                (
                    DIST_LABELS[dist_name],
                    n,
                    Q_LABELS[float(q_level)],
                    f"{method_vars['ECDF']:.8f}",
                    f"{method_vars['Histogram']:.8f}",
                    f"{method_vars['KDE']:.8f}",
                    best_method,
                )
            )

print_table(
    variance_rows,
    headers=["Распределение", "n", "Квантиль", "Var(ECDF)", "Var(Histogram)", "Var(KDE)", "Лучшая"],
)


# %% [markdown]
# ## Выводы по пункту 4
#
# Ниже итог в самой простой форме:
# - если дисперсия меньше, то метод устойчивее;
# - если дисперсия больше, то метод шумнее;
# - хвостовые квантили (`1%`, `5%`) почти всегда оцениваются труднее, чем медиана `50%`.

# %%
print("Сколько раз метод дал минимальную дисперсию среди 24 комбинаций:")
for method_name in ("ECDF", "Histogram", "KDE"):
    print(f"  {method_name:9s}: {best_method_counts[method_name]}")

print("")
print("Простые выводы:")
print("1. Пункт 4 выполнен полностью: для каждого распределения и каждого n сделано 1000 повторов.")
print("2. Для каждого повтора заново считались квантили 1%, 5% и 50% тремя методами.")
print("3. После этого для каждой квантили и каждого метода посчитана выборочная дисперсия.")
print("4. При росте n дисперсии уменьшаются: это значит, что большие выборки дают более стабильные оценки.")
print("5. Для медианы 50% дисперсии обычно меньше, чем для 1% и 5%, потому что центр распределения оценивать легче, чем хвосты.")
print("6. На нормальном распределении KDE часто выигрывает по дисперсии, особенно при больших n.")
print("7. На равномерном распределении гистограмма нередко хорошо работает для хвостовых квантилей, а KDE может страдать из-за сглаживания у границы [0, 1].")
