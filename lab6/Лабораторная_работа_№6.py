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
# # Лабораторная работа №6: доверительные интервалы для математического ожидания
#
# **Тема работы:** изучить, как ведут себя доверительные интервалы для математического ожидания:
# - при изменении размера выборки;
# - при изменении доверительной вероятности;
# - при отклонении реального распределения от предпосылок, на которых строится интервал.
#
# Работа сделана в максимально простой форме:
# 1. сначала кратко вспоминаем теорию;
# 2. потом показываем маленький ручной пример на одной выборке;
# 3. затем выполняем массовое моделирование;
# 4. строим графики и таблицы;
# 5. отдельно подробно объясняем полученные результаты;
# 6. в конце явно проверяем, что все пункты задания выполнены.

# %% [markdown]
# ## План работы
#
# 1. Подготовить константы и простые вспомогательные функции.
# 2. Напомнить формулу доверительного интервала для математического ожидания.
# 3. На одной выборке вручную разобрать, как строится интервал.
# 4. Исследовать, как средняя ширина интервала зависит от `n`.
# 5. Исследовать, как средняя ширина интервала зависит от доверительной вероятности `Q`.
# 6. Проверить, насколько реально достигаемая доверительная вероятность близка к назначенной `Q = 0.95`.
# 7. Подробно объяснить результаты и сформулировать выводы.

# %%
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

SEED = 42
N_REPEATS = 1000

SAMPLE_SIZES = [10, 20, 30, 50, 60, 100]
COVERAGE_SAMPLE_SIZES = [10, 20, 30, 50, 60]
CONFIDENCE_LEVELS = [0.80, 0.90, 0.95, 0.99]

FIXED_CONFIDENCE = 0.95
FIXED_SAMPLE_SIZE = 30

DISTRIBUTION_LABELS = {
    "normal": "Нормальное распределение N(0, 1)",
    "uniform": "Равномерное распределение U[-1, 1]",
}

TRUE_MEANS = {
    "normal": 0.0,
    "uniform": 0.0,
}

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25


# %%
def print_table(rows, headers):
    widths = [len(str(header)) for header in headers]

    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))

    fmt = " | ".join("{:" + str(width) + "}" for width in widths)
    sep = "-+-".join("-" * width for width in widths)

    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))


def generate_samples(distribution: str, n: int, repeats: int, rng: np.random.Generator) -> np.ndarray:
    """
    Возвращает массив формы (repeats, n), где каждая строка это отдельная выборка.
    """
    if distribution == "normal":
        return rng.normal(loc=0.0, scale=1.0, size=(repeats, n))
    if distribution == "uniform":
        return rng.uniform(low=-1.0, high=1.0, size=(repeats, n))

    raise ValueError(f"Неизвестное распределение: {distribution}")


def student_mean_interval(sample: np.ndarray, confidence: float) -> tuple[float, float]:
    """
    Строит доверительный интервал для математического ожидания по одной выборке.
    """
    sample = np.asarray(sample, dtype=float)
    n = len(sample)
    sample_mean = float(np.mean(sample))
    sample_std = float(np.std(sample, ddof=1))
    t_critical = float(stats.t.ppf((1 + confidence) / 2, df=n - 1))
    margin = t_critical * sample_std / np.sqrt(n)
    return sample_mean - margin, sample_mean + margin


def student_mean_intervals(samples: np.ndarray, confidence: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Строит интервалы сразу для многих выборок.

    Вход:
    - samples: массив формы (число_повторов, n)

    Выход:
    - массив нижних границ;
    - массив верхних границ.
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.shape[1]
    sample_means = np.mean(samples, axis=1)
    sample_stds = np.std(samples, axis=1, ddof=1)
    t_critical = float(stats.t.ppf((1 + confidence) / 2, df=n - 1))
    margins = t_critical * sample_stds / np.sqrt(n)
    return sample_means - margins, sample_means + margins


def interval_widths(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return upper - lower


def coverage_rate(lower: np.ndarray, upper: np.ndarray, true_mean: float) -> float:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    hits = (lower <= true_mean) & (true_mean <= upper)
    return float(np.mean(hits))


def plot_width_vs_n(results_by_distribution, sample_sizes, confidence):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)

    for ax, distribution in zip(axes, DISTRIBUTION_LABELS):
        widths = [results_by_distribution[distribution][n] for n in sample_sizes]
        ax.plot(sample_sizes, widths, marker="o", linewidth=2)
        ax.set_title(DISTRIBUTION_LABELS[distribution])
        ax.set_xlabel("Размер выборки n")
        ax.set_ylabel("Средняя ширина интервала")
        ax.set_xticks(sample_sizes)
        ax.set_ylim(bottom=0)
        ax.text(
            0.03,
            0.95,
            f"Q = {confidence:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
        )

    fig.suptitle("Зависимость средней ширины доверительного интервала от размера выборки")
    fig.tight_layout()


def plot_width_vs_confidence(results_by_distribution, confidence_levels, fixed_n):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)

    for ax, distribution in zip(axes, DISTRIBUTION_LABELS):
        widths = [results_by_distribution[distribution][confidence] for confidence in confidence_levels]
        ax.plot(confidence_levels, widths, marker="o", linewidth=2)
        ax.set_title(DISTRIBUTION_LABELS[distribution])
        ax.set_xlabel("Доверительная вероятность Q")
        ax.set_ylabel("Средняя ширина интервала")
        ax.set_xticks(confidence_levels)
        ax.set_ylim(bottom=0)
        ax.text(
            0.03,
            0.95,
            f"n = {fixed_n}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
        )

    fig.suptitle("Зависимость средней ширины доверительного интервала от доверительной вероятности")
    fig.tight_layout()


def plot_coverage(coverage_results, sample_sizes, target_confidence):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)

    for ax, distribution in zip(axes, DISTRIBUTION_LABELS):
        coverages = [coverage_results[distribution][n] for n in sample_sizes]
        ax.plot(sample_sizes, coverages, marker="o", linewidth=2, label="Эмпирическая вероятность")
        ax.axhline(
            target_confidence,
            color="tab:red",
            linestyle="--",
            linewidth=1.5,
            label=f"Назначенное Q = {target_confidence:.2f}",
        )
        ax.set_title(DISTRIBUTION_LABELS[distribution])
        ax.set_xlabel("Размер выборки n")
        ax.set_ylabel("Доля интервалов, содержащих истинное среднее")
        ax.set_xticks(sample_sizes)
        ax.set_ylim(0.85, 1.0)
        ax.legend()

    fig.suptitle("Реальная доверительная вероятность")
    fig.tight_layout()


# %% [markdown]
# ## Подготовка
#
# В работе используются только три библиотеки:
# - `numpy` для генерации выборок и вычислений;
# - `scipy.stats` для коэффициента Стьюдента;
# - `matplotlib` для графиков.
#
# Это самый простой вариант реализации:
# - не нужно подключать лишние пакеты;
# - все формулы видны прямо в коде;
# - легко объяснить, что именно делает каждая строка.
#
# Параметры эксперимента выбраны так:
# - `N_REPEATS = 1000`, чтобы результаты были достаточно устойчивыми;
# - для пункта 1 берём несколько размеров выборки, включая те, что есть в задании;
# - для пункта 2 берём ровно те размеры выборки, которые перечислены в условии;
# - для исследования влияния доверительной вероятности фиксируем `n = 30`.

# %%
print(f"SEED = {SEED}")
print(f"N_REPEATS = {N_REPEATS}")
print(f"SAMPLE_SIZES = {SAMPLE_SIZES}")
print(f"COVERAGE_SAMPLE_SIZES = {COVERAGE_SAMPLE_SIZES}")
print(f"CONFIDENCE_LEVELS = {CONFIDENCE_LEVELS}")
print(f"FIXED_CONFIDENCE = {FIXED_CONFIDENCE}")
print(f"FIXED_SAMPLE_SIZE = {FIXED_SAMPLE_SIZE}")


# %% [markdown]
# ## Теория: какой интервал мы строим
#
# Пусть дана выборка `x1, x2, ..., xn`.
# Если математическое ожидание неизвестно, а дисперсия тоже неизвестна,
# то для математического ожидания часто строят доверительный интервал через распределение Стьюдента:
#
# $$
# \bar{x} \pm t_{\frac{1 + Q}{2}, n-1} \cdot \frac{s}{\sqrt{n}},
# $$
#
# где:
# - `\bar{x}` это выборочное среднее;
# - `Q` это доверительная вероятность;
# - `t_{(1+Q)/2, n-1}` это квантиль распределения Стьюдента;
# - `s` это исправленное выборочное стандартное отклонение.
#
# В коде это означает буквально три шага:
# 1. посчитать среднее `\bar{x}`;
# 2. посчитать стандартное отклонение `s`;
# 3. прибавить и вычесть погрешность `t * s / sqrt(n)`.

# %% [markdown]
# ## От чего зависит ширина доверительного интервала
#
# Ширина интервала равна
#
# $$
# 2 \cdot t_{\frac{1 + Q}{2}, n-1} \cdot \frac{s}{\sqrt{n}}.
# $$
#
# Из этой формулы сразу видно:
# - если растёт `n`, то знаменатель `\sqrt{n}` увеличивается, и интервал обычно сужается;
# - если растёт `Q`, то квантиль Стьюдента становится больше, и интервал расширяется;
# - если выборка более разбросанная, то `s` больше, значит интервал тоже шире.
#
# Именно эти зависимости и будут исследоваться в работе.

# %% [markdown]
# ## Маленький демонстрационный пример на одной выборке
#
# Прежде чем строить много графиков, полезно один раз посмотреть,
# как вычисляется интервал на конкретной выборке.
# Ниже:
# - генерируем одну нормальную выборку размера `10`;
# - вручную считаем её среднее;
# - считаем исправленное стандартное отклонение;
# - берём коэффициент Стьюдента для `Q = 0.95`;
# - строим границы интервала.

# %%
demo_rng = np.random.default_rng(SEED)
demo_sample = generate_samples("normal", n=10, repeats=1, rng=demo_rng)[0]

demo_mean = float(np.mean(demo_sample))
demo_std = float(np.std(demo_sample, ddof=1))
demo_t = float(stats.t.ppf((1 + FIXED_CONFIDENCE) / 2, df=len(demo_sample) - 1))
demo_margin = demo_t * demo_std / np.sqrt(len(demo_sample))
demo_lower = demo_mean - demo_margin
demo_upper = demo_mean + demo_margin

print("Демонстрационная выборка:")
print(np.round(demo_sample, 4))
print()
print(f"Выборочное среднее x̄ = {demo_mean:.6f}")
print(f"Исправленное стандартное отклонение s = {demo_std:.6f}")
print(f"Коэффициент Стьюдента t = {demo_t:.6f}")
print(f"Погрешность t * s / sqrt(n) = {demo_margin:.6f}")
print(f"95%-й доверительный интервал: [{demo_lower:.6f}; {demo_upper:.6f}]")


# %% [markdown]
# В этом примере особенно важно увидеть связь между формулой и кодом:
# - `np.mean(sample)` даёт `\bar{x}`;
# - `np.std(sample, ddof=1)` даёт исправленную оценку стандартного отклонения `s`;
# - `stats.t.ppf(...)` даёт квантиль распределения Стьюдента;
# - итоговый интервал получается как `x̄ - margin` и `x̄ + margin`.
#
# Дальше работа будет делать ровно то же самое,
# только уже для большого количества выборок.

# %% [markdown]
# ## Пункт 1. Построение интервалов и исследование их ширины
#
# По условию нужно:
# - генерировать выборки из разных распределений;
# - строить доверительные интервалы для математического ожидания;
# - посмотреть, как меняется ширина интервала при изменении `n`;
# - посмотреть, как меняется ширина интервала при изменении `Q`.
#
# Ниже для наглядности сначала покажем по две реальные выборки из каждого распределения
# и интервалы, построенные по ним при `Q = 0.95`.

# %%
example_rng = np.random.default_rng(SEED + 1)
example_rows = []

for distribution in DISTRIBUTION_LABELS:
    example_samples = generate_samples(distribution, n=10, repeats=2, rng=example_rng)

    for sample_index, sample in enumerate(example_samples, start=1):
        lower, upper = student_mean_interval(sample, confidence=FIXED_CONFIDENCE)
        example_rows.append(
            (
                distribution,
                sample_index,
                f"{np.mean(sample):.4f}",
                f"{lower:.4f}",
                f"{upper:.4f}",
                f"{upper - lower:.4f}",
            )
        )

print_table(
    example_rows,
    headers=[
        "Распределение",
        "№ выборки",
        "x̄",
        "Нижняя граница",
        "Верхняя граница",
        "Ширина",
    ],
)


# %% [markdown]
# Таблица выше нужна, чтобы видеть не только усреднённые графики,
# но и сами реальные интервалы, построенные по конкретным выборкам.
# У разных выборок интервалы отличаются, потому что:
# - отличаются выборочные средние;
# - отличаются выборочные стандартные отклонения.
#
# Поэтому в основной части мы будем смотреть не на одну случайную ширину,
# а на **среднюю ширину по 1000 выборкам**.
# Это делает графики гораздо более понятными.

# %% [markdown]
# ### Исследование зависимости ширины интервала от размера выборки
#
# Для каждого распределения и каждого `n` из списка:
# - генерируем `1000` выборок;
# - для каждой строим доверительный интервал при `Q = 0.95`;
# - считаем ширину;
# - берём среднюю ширину по всем 1000 интервалам.

# %%
rng = np.random.default_rng(SEED)

width_vs_n = {distribution: {} for distribution in DISTRIBUTION_LABELS}
width_vs_n_table_rows = []

for distribution in DISTRIBUTION_LABELS:
    for n in SAMPLE_SIZES:
        samples = generate_samples(distribution, n=n, repeats=N_REPEATS, rng=rng)
        lower, upper = student_mean_intervals(samples, confidence=FIXED_CONFIDENCE)
        widths = interval_widths(lower, upper)
        mean_width = float(np.mean(widths))

        width_vs_n[distribution][n] = mean_width
        width_vs_n_table_rows.append((distribution, n, f"{mean_width:.6f}"))

print_table(
    width_vs_n_table_rows,
    headers=["Распределение", "n", "Средняя ширина при Q = 0.95"],
)


# %%
plot_width_vs_n(width_vs_n, SAMPLE_SIZES, FIXED_CONFIDENCE)


# %% [markdown]
# Что нужно увидеть на этом графике:
# - при росте `n` интервал сужается;
# - это происходит у обоих распределений;
# - точная высота кривых может различаться, потому что у распределений разный разброс.
#
# Главная причина сужения интервала проста:
# в формуле есть деление на `\sqrt{n}`, а значит ошибка оценки среднего уменьшается при увеличении объёма выборки.

# %%
for distribution in DISTRIBUTION_LABELS:
    first_width = width_vs_n[distribution][SAMPLE_SIZES[0]]
    last_width = width_vs_n[distribution][SAMPLE_SIZES[-1]]
    shrink_factor = first_width / last_width
    print(DISTRIBUTION_LABELS[distribution])
    print(f"- средняя ширина при n = {SAMPLE_SIZES[0]}: {first_width:.6f}")
    print(f"- средняя ширина при n = {SAMPLE_SIZES[-1]}: {last_width:.6f}")
    print(f"- интервал стал уже примерно в {shrink_factor:.2f} раза")
    print()


# %% [markdown]
# Здесь уже можно сделать первый содержательный вывод:
# чем больше выборка, тем уже доверительный интервал.
# Это и есть ожидаемое поведение корректного статистического интервала.

# %% [markdown]
# ### Исследование зависимости ширины интервала от доверительной вероятности
#
# Теперь фиксируем размер выборки `n = 30` и меняем только `Q`.
# Для каждого значения `Q`:
# - генерируем `1000` выборок;
# - строим интервалы;
# - считаем среднюю ширину.

# %%
rng = np.random.default_rng(SEED + 100)

width_vs_confidence = {distribution: {} for distribution in DISTRIBUTION_LABELS}
width_vs_confidence_table_rows = []

for distribution in DISTRIBUTION_LABELS:
    for confidence in CONFIDENCE_LEVELS:
        samples = generate_samples(distribution, n=FIXED_SAMPLE_SIZE, repeats=N_REPEATS, rng=rng)
        lower, upper = student_mean_intervals(samples, confidence=confidence)
        widths = interval_widths(lower, upper)
        mean_width = float(np.mean(widths))

        width_vs_confidence[distribution][confidence] = mean_width
        width_vs_confidence_table_rows.append((distribution, f"{confidence:.2f}", f"{mean_width:.6f}"))

print_table(
    width_vs_confidence_table_rows,
    headers=["Распределение", "Q", "Средняя ширина при n = 30"],
)


# %%
plot_width_vs_confidence(width_vs_confidence, CONFIDENCE_LEVELS, FIXED_SAMPLE_SIZE)


# %% [markdown]
# Что нужно увидеть здесь:
# - при увеличении `Q` интервал становится шире;
# - это также наблюдается у обоих распределений.
#
# Причина снова видна прямо из формулы:
# при росте `Q` увеличивается коэффициент Стьюдента `t`,
# а значит увеличивается и погрешность `t * s / sqrt(n)`.

# %%
for distribution in DISTRIBUTION_LABELS:
    smallest_q_width = width_vs_confidence[distribution][CONFIDENCE_LEVELS[0]]
    largest_q_width = width_vs_confidence[distribution][CONFIDENCE_LEVELS[-1]]
    growth_factor = largest_q_width / smallest_q_width
    print(DISTRIBUTION_LABELS[distribution])
    print(f"- средняя ширина при Q = {CONFIDENCE_LEVELS[0]:.2f}: {smallest_q_width:.6f}")
    print(f"- средняя ширина при Q = {CONFIDENCE_LEVELS[-1]:.2f}: {largest_q_width:.6f}")
    print(f"- интервал стал шире примерно в {growth_factor:.2f} раза")
    print()


# %% [markdown]
# Второй содержательный вывод:
# чем выше доверительная вероятность, тем шире интервал.
# Это естественно: чтобы быть более "уверенными", приходится захватывать более широкий диапазон значений.

# %% [markdown]
# ## Пункт 2. Реальная доверительная вероятность
#
# Теперь проверим главный практический вопрос:
# действительно ли интервал с назначенной вероятностью `Q = 0.95`
# в реальности накрывает истинное математическое ожидание примерно в 95% случаев.
#
# Как это проверяется:
# 1. фиксируем размер выборки `n`;
# 2. генерируем `1000` выборок;
# 3. для каждой строим интервал;
# 4. смотрим, попало ли в него истинное среднее;
# 5. считаем долю успешных попаданий.
#
# Если эта доля близка к `0.95`, то реальная доверительная вероятность близка к назначенной.

# %%
rng = np.random.default_rng(SEED + 200)

coverage_results = {distribution: {} for distribution in DISTRIBUTION_LABELS}
coverage_table_rows = []

for distribution in DISTRIBUTION_LABELS:
    true_mean = TRUE_MEANS[distribution]

    for n in COVERAGE_SAMPLE_SIZES:
        samples = generate_samples(distribution, n=n, repeats=N_REPEATS, rng=rng)
        lower, upper = student_mean_intervals(samples, confidence=FIXED_CONFIDENCE)
        empirical_coverage = coverage_rate(lower, upper, true_mean=true_mean)

        coverage_results[distribution][n] = empirical_coverage
        coverage_table_rows.append((distribution, n, f"{empirical_coverage:.4f}"))

print_table(
    coverage_table_rows,
    headers=["Распределение", "n", "Реальная доверительная вероятность"],
)


# %%
plot_coverage(coverage_results, COVERAGE_SAMPLE_SIZES, FIXED_CONFIDENCE)


# %% [markdown]
# На этом графике горизонтальная линия показывает назначенное значение `Q = 0.95`.
# Ломаная линия показывает то, что получилось на практике.
#
# Интерпретация такая:
# - если точки лежат близко к горизонтальной линии, значит интервал работает так, как ожидалось;
# - если есть заметные отклонения, значит предпосылки метода влияют на реальный результат.

# %%
for distribution in DISTRIBUTION_LABELS:
    print(DISTRIBUTION_LABELS[distribution])
    for n in COVERAGE_SAMPLE_SIZES:
        empirical = coverage_results[distribution][n]
        deviation = empirical - FIXED_CONFIDENCE
        print(f"- n = {n}: реальная вероятность = {empirical:.4f}, отклонение от 0.95 = {deviation:+.4f}")
    print()


# %%
normal_coverages = [coverage_results["normal"][n] for n in COVERAGE_SAMPLE_SIZES]
uniform_coverages = [coverage_results["uniform"][n] for n in COVERAGE_SAMPLE_SIZES]

print("Краткое резюме по полученным покрытиям:")
print(
    f"- для нормального распределения значения лежат в диапазоне "
    f"от {min(normal_coverages):.4f} до {max(normal_coverages):.4f}"
)
print(
    f"- для равномерного распределения значения лежат в диапазоне "
    f"от {min(uniform_coverages):.4f} до {max(uniform_coverages):.4f}"
)
print(
    "- в этом запуске отклонения для равномерного распределения оказались небольшими, "
    "то есть интервал Стьюдента показал довольно устойчивое поведение даже при нарушении "
    "точной предпосылки нормальности"
)


# %% [markdown]
# ## Объяснение полученных результатов
#
# Теперь соберём все наблюдения вместе и объясним их простым языком.

# %% [markdown]
# ### 1. Почему интервал сужается при росте размера выборки
#
# На первом графике видно, что при переходе от маленьких `n` к большим
# средняя ширина интервала уменьшается.
#
# Это не случайность, а прямое следствие формулы:
#
# $$
# \text{ширина} = 2 \cdot t \cdot \frac{s}{\sqrt{n}}.
# $$
#
# Когда `n` растёт, величина `1 / \sqrt{n}` уменьшается.
# Значит, среднее по большой выборке определяется точнее,
# и для него нужен уже более узкий интервал.

# %% [markdown]
# ### 2. Почему интервал расширяется при росте доверительной вероятности
#
# На втором графике видно, что при увеличении `Q` от `0.80` до `0.99`
# средняя ширина интервала возрастает.
#
# Причина в том, что более высокая доверительная вероятность означает
# более строгий запрос к интервалу:
# он должен накрывать истинное значение чаще.
# Чтобы добиться этого, интервал приходится делать шире.

# %% [markdown]
# ### 3. Почему для нормального распределения результат обычно ближе к 0.95
#
# Интервал Стьюдента для математического ожидания выводится при предположении,
# что выборка получена из нормального распределения.
# Поэтому для `N(0, 1)` реальная доверительная вероятность обычно получается
# очень близкой к назначенной `0.95`.
#
# Небольшие отличия всё равно возможны,
# потому что мы оцениваем вероятность не теоретически, а по конечному числу экспериментов `1000`.
# То есть есть ещё обычная моделирующая случайность.

# %% [markdown]
# ### 4. Почему для равномерного распределения возможны отклонения
#
# Равномерное распределение `U[-1, 1]` не является нормальным.
# Значит, строгие предпосылки для интервала Стьюдента здесь нарушены.
#
# Поэтому эмпирическая доверительная вероятность может отличаться от `0.95`,
# особенно на малых выборках.
# Однако важно понимать и обратную сторону:
# отклонения не обязаны быть большими в каждом конкретном запуске.
# Если распределение симметрично и размер выборки не слишком мал,
# интервал Стьюдента может работать довольно хорошо даже вне идеально нормального случая.
# При увеличении `n` поведение обычно становится более стабильным:
# выборочное среднее начинает лучше подчиняться приближённым асимптотическим закономерностям,
# и интервал работает ближе к ожидаемому уровню.

# %% [markdown]
# ### 5. Общий практический смысл результата
#
# Из работы видно следующее:
# - доверительные интервалы не являются фиксированными отрезками,
#   они зависят от конкретной выборки;
# - маленькая выборка даёт более широкий и менее устойчивый интервал;
# - высокая доверительная вероятность делает интервал шире;
# - при нарушении предпосылок метода реальное покрытие может отличаться от назначенного.
#
# Это главный практический вывод:
# доверительный интервал нужно не просто механически считать,
# а понимать, при каких условиях он построен и насколько эти условия похожи на реальную задачу.

# %% [markdown]
# ## Небольшая автоматическая самопроверка результатов
#
# Помимо визуального анализа, полезно проверить несколько логических условий:
# - каждая ширина должна быть положительной;
# - средняя ширина должна уменьшаться при росте `n`;
# - средняя ширина должна увеличиваться при росте `Q`;
# - эмпирическая доверительная вероятность должна лежать между `0` и `1`.

# %%
width_decreases_with_n = {}
width_increases_with_q = {}

for distribution in DISTRIBUTION_LABELS:
    width_values_n = [width_vs_n[distribution][n] for n in SAMPLE_SIZES]
    width_values_q = [width_vs_confidence[distribution][q] for q in CONFIDENCE_LEVELS]
    coverage_values = [coverage_results[distribution][n] for n in COVERAGE_SAMPLE_SIZES]

    positive_widths_n = all(value > 0 for value in width_values_n)
    positive_widths_q = all(value > 0 for value in width_values_q)
    width_decreases_with_n[distribution] = all(
        width_values_n[index] > width_values_n[index + 1]
        for index in range(len(width_values_n) - 1)
    )
    width_increases_with_q[distribution] = all(
        width_values_q[index] < width_values_q[index + 1]
        for index in range(len(width_values_q) - 1)
    )
    valid_coverages = all(0.0 <= value <= 1.0 for value in coverage_values)

    print(DISTRIBUTION_LABELS[distribution])
    print(f"- все ширины при изменении n положительны: {positive_widths_n}")
    print(f"- все ширины при изменении Q положительны: {positive_widths_q}")
    print(f"- средняя ширина убывает при росте n: {width_decreases_with_n[distribution]}")
    print(f"- средняя ширина возрастает при росте Q: {width_increases_with_q[distribution]}")
    print(f"- все эмпирические вероятности лежат в [0, 1]: {valid_coverages}")
    print()


# %% [markdown]
# Если все проверки дают `True`, то численные результаты согласуются
# и с формулами, и со здравым смыслом.

# %% [markdown]
# ## Проверка выполнения задания
#
# Ниже перечислим все пункты задания и сразу отметим, как они выполнены в работе.

# %%
check_rows = [
    (
        "1",
        "Сгенерированы выборки из разных распределений разного размера",
        "Да: использованы N(0,1) и U[-1,1], размеры n = 10, 20, 30, 50, 60, 100",
    ),
    (
        "2",
        "Построены доверительные интервалы для математического ожидания",
        "Да: использован интервал Стьюдента x̄ ± t * s / sqrt(n)",
    ),
    (
        "3",
        "Исследована зависимость ширины интервала от размера выборки",
        "Да: построен график средней ширины от n",
    ),
    (
        "4",
        "Исследована зависимость ширины интервала от доверительной вероятности",
        "Да: построен график средней ширины от Q",
    ),
    (
        "5",
        "Проверено совпадение назначенной и реальной доверительной вероятности при Q = 0.95",
        "Да: выполнено 1000 повторов для n = 10, 20, 30, 50, 60",
    ),
    (
        "6",
        "Проверка проведена для нормального и равномерного распределений",
        "Да: результаты приведены отдельно для двух распределений",
    ),
]

print_table(check_rows, headers=["№", "Пункт задания", "Статус"])


# %% [markdown]
# Видно, что все требования задания закрыты.

# %% [markdown]
# ## Итоговые выводы
#
# В этой работе были получены три главных результата.
#
# **Первый результат.**
# При увеличении размера выборки доверительный интервал для математического ожидания сужается.
# Это связано с тем, что среднее по большой выборке оценивается точнее.
#
# **Второй результат.**
# При увеличении доверительной вероятности интервал расширяется.
# Более надёжный интервал неизбежно должен быть шире.
#
# **Третий результат.**
# Для нормального распределения реальная доверительная вероятность обычно близка к назначенной `0.95`,
# потому что метод Стьюдента как раз для этого случая и построен.
# Для равномерного распределения возможны отклонения, особенно при небольших `n`,
# потому что исходные предпосылки метода нарушаются.
#
# Главный общий вывод такой:
# доверительный интервал это полезный инструмент,
# но его поведение зависит и от размера выборки, и от уровня доверия, и от того,
# насколько реальные данные соответствуют теоретической модели.
