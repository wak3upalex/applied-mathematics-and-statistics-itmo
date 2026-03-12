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
# # Лабораторная работа №5: сравнение статистических свойств точечных оценок математического ожидания и дисперсии
#
# **Цель работы:** на простом и полностью воспроизводимом моделировании посмотреть,
# как ведут себя три оценки:
# - выборочное среднее `c* = \bar{x}`;
# - неисправленная оценка дисперсии;
# - исправленная оценка дисперсии.
#
# Важно не просто получить графики, а правильно их интерпретировать:
# - где оценка смещённая, а где несмещённая;
# - как меняются дисперсии оценок при росте `n`;
# - насколько хорошо эмпирические распределения оценок совпадают с теоретическими.
#
# Вся реализация будет максимально прямой:
# 1. генерируем много выборок из `N(0, 1)`;
# 2. для каждой выборки считаем нужные оценки;
# 3. по множеству полученных оценок считаем эмпирические средние и дисперсии;
# 4. сравниваем их с теорией;
# 5. отдельно сравниваем эмпирические и теоретические функции распределения.

# %% [markdown]
# ## План ноутбука
#
# 1. Подготовим константы и вспомогательные функции.
# 2. Поясним теорию для трёх оценок.
# 3. На одном маленьком примере покажем, что именно считается внутри формул.
# 4. Выполним массовое моделирование для `n = 10, 20, ..., 100`.
# 5. Построим графики эмпирических и теоретических средних и дисперсий.
# 6. Для `n = 10` и `n = 100` сравним эмпирические функции распределения с теоретическими.
# 7. Проверим, что все пункты задания выполнены.
# 8. Сформулируем выводы простым языком.

# %%
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

SEED = 42
N_REPEATS = 1000
SAMPLE_SIZES = np.arange(10, 101, 10)

TRUE_MEAN = 0.0
TRUE_VARIANCE = 1.0
TRUE_STD = np.sqrt(TRUE_VARIANCE)

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


def generate_samples(n, repeats, rng):
    """
    Генерирует массив формы (repeats, n) из нормального распределения N(0, 1).

    Каждая строка — это отдельная выборка размера n.
    """
    return rng.normal(loc=TRUE_MEAN, scale=TRUE_STD, size=(repeats, n))


def sample_mean_estimates(samples):
    """Возвращает выборочные средние для всех сгенерированных выборок."""
    return np.mean(samples, axis=1)


def uncorrected_variance_estimates(samples):
    """
    Неисправленная оценка дисперсии:
    D_n^* = (1 / n) * sum((x_i - x_bar)^2).
    """
    sample_means = sample_mean_estimates(samples)
    centered = samples - sample_means[:, None]
    return np.mean(centered**2, axis=1)


def corrected_variance_estimates(samples):
    """
    Исправленная оценка дисперсии:
    S^2 = (1 / (n - 1)) * sum((x_i - x_bar)^2).
    """
    n = samples.shape[1]
    sample_means = sample_mean_estimates(samples)
    centered = samples - sample_means[:, None]
    return np.sum(centered**2, axis=1) / (n - 1)


def summarize_estimates(estimates):
    """
    По массиву оценок считаем:
    - эмпирическое среднее;
    - эмпирическую дисперсию по 1000 повторным экспериментам.

    Для эмпирической дисперсии используем ddof=1, то есть обычную
    выборочную оценку дисперсии по набору полученных оценок.
    """
    estimates = np.asarray(estimates, dtype=float)
    empirical_mean = float(np.mean(estimates))
    empirical_variance = float(np.var(estimates, ddof=1))
    return empirical_mean, empirical_variance


def ecdf(values):
    """
    Строит выборочную функцию распределения.

    Возвращает:
    - отсортированные значения;
    - уровни ступенек 1/n, 2/n, ..., n/n.
    """
    sorted_values = np.sort(np.asarray(values, dtype=float))
    n = len(sorted_values)
    probs = np.arange(1, n + 1) / n
    return sorted_values, probs


def theoretical_mean_stats(n):
    return TRUE_MEAN, TRUE_VARIANCE / n


def theoretical_uncorrected_var_stats(n):
    theoretical_mean = TRUE_VARIANCE * (n - 1) / n
    theoretical_variance = 2 * (TRUE_VARIANCE**2) * (n - 1) / (n**2)
    return theoretical_mean, theoretical_variance


def theoretical_corrected_var_stats(n):
    theoretical_mean = TRUE_VARIANCE
    theoretical_variance = 2 * (TRUE_VARIANCE**2) / (n - 1)
    return theoretical_mean, theoretical_variance


def theoretical_cdf_mean(x, n):
    x = np.asarray(x, dtype=float)
    return stats.norm.cdf(x, loc=TRUE_MEAN, scale=np.sqrt(TRUE_VARIANCE / n))


def theoretical_cdf_uncorrected_var(x, n):
    x = np.asarray(x, dtype=float)
    clipped = np.maximum(x, 0.0)
    return stats.chi2.cdf(n * clipped / TRUE_VARIANCE, df=n - 1)


def theoretical_cdf_corrected_var(x, n):
    x = np.asarray(x, dtype=float)
    clipped = np.maximum(x, 0.0)
    return stats.chi2.cdf((n - 1) * clipped / TRUE_VARIANCE, df=n - 1)


def run_experiment_for_n(n, repeats, rng):
    samples = generate_samples(n, repeats, rng)

    mean_estimates = sample_mean_estimates(samples)
    uncorrected_var_estimates = uncorrected_variance_estimates(samples)
    corrected_var_estimates = corrected_variance_estimates(samples)

    return {
        "samples": samples,
        "mean_estimates": mean_estimates,
        "uncorrected_var_estimates": uncorrected_var_estimates,
        "corrected_var_estimates": corrected_var_estimates,
    }


def plot_stats_comparison(
    sample_sizes,
    empirical_means,
    theoretical_means,
    empirical_variances,
    theoretical_variances,
    title_prefix,
    mean_label,
    variance_label,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    axes[0].plot(sample_sizes, empirical_means, marker="o", label="Эмпирическое среднее")
    axes[0].plot(sample_sizes, theoretical_means, marker="s", label="Теоретическое среднее")
    axes[0].set_title(f"{title_prefix}: среднее")
    axes[0].set_xlabel("Размер выборки n")
    axes[0].set_ylabel(mean_label)
    axes[0].legend()

    axes[1].plot(sample_sizes, empirical_variances, marker="o", label="Эмпирическая дисперсия")
    axes[1].plot(sample_sizes, theoretical_variances, marker="s", label="Теоретическая дисперсия")
    axes[1].set_title(f"{title_prefix}: дисперсия")
    axes[1].set_xlabel("Размер выборки n")
    axes[1].set_ylabel(variance_label)
    axes[1].legend()

    fig.tight_layout()


def plot_ecdf_vs_theory(ax, estimates, cdf_function, n, title):
    x_empirical, y_empirical = ecdf(estimates)
    x_min = float(np.min(estimates))
    x_max = float(np.max(estimates))
    span = x_max - x_min
    padding = 0.08 * span if span > 0 else 0.5
    grid = np.linspace(x_min - padding, x_max + padding, 500)

    ax.step(x_empirical, y_empirical, where="post", label="Эмпирическая CDF")
    ax.plot(grid, cdf_function(grid, n), label="Теоретическая CDF")
    ax.set_title(title)
    ax.set_xlabel("Значение оценки")
    ax.set_ylabel("F(x)")
    ax.legend()


# %% [markdown]
# ## Подготовка
#
# В работе используем только три стандартных инструмента:
# - `numpy` для генерации выборок и вычислений;
# - `matplotlib` для графиков;
# - `scipy.stats` для теоретических функций распределения.
#
# Параметры эксперимента такие:
# - истинное математическое ожидание равно `0`;
# - истинная дисперсия равна `1`;
# - число повторений для каждого `n` равно `1000`;
# - размеры выборки: `10, 20, ..., 100`.
#
# Один фиксированный `SEED` нужен для воспроизводимости:
# при повторном запуске будут получаться те же самые результаты.

# %%
print(f"SEED = {SEED}")
print(f"N_REPEATS = {N_REPEATS}")
print(f"SAMPLE_SIZES = {SAMPLE_SIZES.tolist()}")
print(f"TRUE_MEAN = {TRUE_MEAN}")
print(f"TRUE_VARIANCE = {TRUE_VARIANCE}")


# %% [markdown]
# ## Теория: что именно мы сравниваем
#
# По условию мы много раз генерируем выборку из нормального распределения `N(0, 1)`.
# Для каждой отдельной выборки считаем три оценки.
#
# ### 1. Выборочное среднее
#
# $$
# c^* = \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i.
# $$
#
# Если исходное распределение нормальное `N(0, 1)`, то:
#
# $$
# E[\bar{X}] = 0, \qquad D[\bar{X}] = \frac{1}{n}, \qquad \bar{X} \sim N\left(0, \frac{1}{n}\right).
# $$
#
# Здесь важно понимать: `\bar{X}` — это случайная величина, потому что сама выборка случайна.
# Значит, если мы `1000` раз пересоздадим выборку и каждый раз пересчитаем `\bar{x}`,
# то получим уже **1000 значений одной и той же оценки**. По этим 1000 значениям
# можно эмпирически изучать её свойства.

# %% [markdown]
# ### 2. Неисправленная оценка дисперсии
#
# $$
# D_n^* = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2.
# $$
#
# Для нормальной выборки справедливо:
#
# $$
# n D_n^* \sim \chi^2_{n-1}.
# $$
#
# Отсюда следуют формулы:
#
# $$
# E[D_n^*] = \frac{n-1}{n},
# \qquad
# D[D_n^*] = \frac{2(n-1)}{n^2}.
# $$
#
# Среднее этой оценки равно не `1`, а `(n-1)/n`.
# Значит, она **смещённая вниз**: в среднем немного занижает истинную дисперсию.

# %% [markdown]
# ### 3. Исправленная оценка дисперсии
#
# $$
# S^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2.
# $$
#
# Для неё:
#
# $$
# (n-1)S^2 \sim \chi^2_{n-1}.
# $$
#
# Поэтому:
#
# $$
# E[S^2] = 1,
# \qquad
# D[S^2] = \frac{2}{n-1}.
# $$
#
# То есть исправленная оценка дисперсии является **несмещённой**:
# её среднее совпадает с истинной дисперсией.

# %% [markdown]
# ## Почему мы считаем среднее и дисперсию по 1000 полученным оценкам
#
# Для каждого фиксированного `n` делаем одинаковую процедуру:
# 1. Генерируем `1000` независимых выборок размера `n`.
# 2. Для каждой выборки считаем интересующую оценку.
# 3. Получаем `1000` значений этой оценки.
#
# Дальше эти `1000` чисел можно рассматривать как большую выборку
# из распределения самой оценки.
#
# Тогда:
# - среднее по этим `1000` значениям приближает теоретическое `E[оценки]`;
# - дисперсия по этим `1000` значениям приближает теоретическое `D[оценки]`;
# - эмпирическая функция распределения приближает теоретическую CDF оценки.
#
# Именно поэтому моделирование позволяет на практике проверить теоретические формулы.

# %% [markdown]
# ## Маленький демонстрационный пример на одной выборке
#
# Прежде чем переходить к массовому моделированию, полезно один раз увидеть,
# что именно считается в формулах.
# Ниже берём одну выборку размера `10` и вручную считаем три оценки.

# %%
demo_rng = np.random.default_rng(SEED)
demo_sample = generate_samples(n=10, repeats=1, rng=demo_rng)[0]

demo_mean = float(np.mean(demo_sample))
demo_uncorrected_var = float(np.mean((demo_sample - demo_mean) ** 2))
demo_corrected_var = float(np.sum((demo_sample - demo_mean) ** 2) / (len(demo_sample) - 1))

print("Одна демонстрационная выборка размера 10:")
print(np.round(demo_sample, 4))
print()
print(f"Выборочное среднее c* = {demo_mean:.6f}")
print(f"Неисправленная дисперсия D_n* = {demo_uncorrected_var:.6f}")
print(f"Исправленная дисперсия S^2 = {demo_corrected_var:.6f}")


# %% [markdown]
# В этом примере особенно важно увидеть связь между формулами и кодом:
# - выборочное среднее это просто `np.mean(sample)`;
# - неисправленная дисперсия это среднее квадратов отклонений от `\bar{x}`;
# - исправленная дисперсия отличается только делением на `n-1` вместо `n`.
#
# Дальше мы будем делать **ровно те же действия**,
# только не для одной выборки, а для `1000` выборок сразу.

# %% [markdown]
# ## Массовое моделирование
#
# Теперь переходим к главной части работы.
# Для каждого `n` из набора `10, 20, ..., 100`:
# - генерируем `1000` выборок;
# - считаем по ним три вида оценок;
# - сохраняем и сами оценки, и их эмпирические характеристики.

# %%
rng = np.random.default_rng(SEED)

results_by_n = {}

mean_empirical_means = []
mean_empirical_variances = []
mean_theoretical_means = []
mean_theoretical_variances = []

unc_empirical_means = []
unc_empirical_variances = []
unc_theoretical_means = []
unc_theoretical_variances = []

cor_empirical_means = []
cor_empirical_variances = []
cor_theoretical_means = []
cor_theoretical_variances = []

summary_rows = []

for n in SAMPLE_SIZES:
    current_results = run_experiment_for_n(n=n, repeats=N_REPEATS, rng=rng)
    results_by_n[int(n)] = current_results

    mean_emp_mean, mean_emp_var = summarize_estimates(current_results["mean_estimates"])
    unc_emp_mean, unc_emp_var = summarize_estimates(current_results["uncorrected_var_estimates"])
    cor_emp_mean, cor_emp_var = summarize_estimates(current_results["corrected_var_estimates"])

    mean_theory_mean, mean_theory_var = theoretical_mean_stats(int(n))
    unc_theory_mean, unc_theory_var = theoretical_uncorrected_var_stats(int(n))
    cor_theory_mean, cor_theory_var = theoretical_corrected_var_stats(int(n))

    mean_empirical_means.append(mean_emp_mean)
    mean_empirical_variances.append(mean_emp_var)
    mean_theoretical_means.append(mean_theory_mean)
    mean_theoretical_variances.append(mean_theory_var)

    unc_empirical_means.append(unc_emp_mean)
    unc_empirical_variances.append(unc_emp_var)
    unc_theoretical_means.append(unc_theory_mean)
    unc_theoretical_variances.append(unc_theory_var)

    cor_empirical_means.append(cor_emp_mean)
    cor_empirical_variances.append(cor_emp_var)
    cor_theoretical_means.append(cor_theory_mean)
    cor_theoretical_variances.append(cor_theory_var)

    if n in (10, 50, 100):
        summary_rows.append(
            (
                int(n),
                f"{mean_emp_mean:.4f}",
                f"{mean_theory_mean:.4f}",
                f"{unc_emp_mean:.4f}",
                f"{unc_theory_mean:.4f}",
                f"{cor_emp_mean:.4f}",
                f"{cor_theory_mean:.4f}",
            )
        )

print_table(
    summary_rows,
    headers=[
        "n",
        "M[c*] эмп.",
        "M[c*] теор.",
        "M[D_n*] эмп.",
        "M[D_n*] теор.",
        "M[S^2] эмп.",
        "M[S^2] теор.",
    ],
)


# %% [markdown]
# В таблице выше показаны только несколько размеров выборки,
# чтобы вывод не был слишком длинным.
# Уже по ней обычно видно главное:
# - среднее выборочного среднего близко к `0`;
# - среднее неисправленной дисперсии близко к `(n-1)/n`;
# - среднее исправленной дисперсии близко к `1`.

# %% [markdown]
# ## Пункт 1. Сравнение эмпирических и теоретических средних и дисперсий
#
# По условию нужно сравнить:
# - эмпирические средние оценок;
# - эмпирические дисперсии оценок;
# - соответствующие теоретические значения.
#
# Ниже делаем это отдельно для каждой оценки.

# %%
plot_stats_comparison(
    sample_sizes=SAMPLE_SIZES,
    empirical_means=mean_empirical_means,
    theoretical_means=mean_theoretical_means,
    empirical_variances=mean_empirical_variances,
    theoretical_variances=mean_theoretical_variances,
    title_prefix="Выборочное среднее",
    mean_label="M[c*]",
    variance_label="D[c*]",
)


# %% [markdown]
# На этих графиках нужно смотреть на две вещи:
# - среднее `c*` должно быть около `0`, значит оценка математического ожидания несмещённая;
# - дисперсия `c*` должна убывать как `1/n`, то есть при росте `n`
#   оценка становится более стабильной.

# %%
plot_stats_comparison(
    sample_sizes=SAMPLE_SIZES,
    empirical_means=unc_empirical_means,
    theoretical_means=unc_theoretical_means,
    empirical_variances=unc_empirical_variances,
    theoretical_variances=unc_theoretical_variances,
    title_prefix="Неисправленная оценка дисперсии",
    mean_label="M[D_n*]",
    variance_label="D[D_n*]",
)


# %% [markdown]
# Здесь главный визуальный признак смещённости такой:
# кривая теоретического среднего и эмпирического среднего лежит ниже `1`,
# особенно заметно при маленьких `n`.
# Это и означает, что неисправленная оценка в среднем занижает дисперсию.

# %%
plot_stats_comparison(
    sample_sizes=SAMPLE_SIZES,
    empirical_means=cor_empirical_means,
    theoretical_means=cor_theoretical_means,
    empirical_variances=cor_empirical_variances,
    theoretical_variances=cor_theoretical_variances,
    title_prefix="Исправленная оценка дисперсии",
    mean_label="M[S^2]",
    variance_label="D[S^2]",
)


# %% [markdown]
# Для исправленной оценки график среднего должен идти около `1`.
# Это визуально подтверждает её несмещённость.
# При этом дисперсия оценки тоже уменьшается при росте `n`,
# то есть оценка становится устойчивее на больших выборках.

# %% [markdown]
# ## Промежуточные выводы по пункту 1
#
# По теории и по моделированию ожидаем:
# - `c* = \bar{x}` — несмещённая оценка математического ожидания;
# - `D_n^*` — смещённая оценка дисперсии;
# - `S^2` — несмещённая оценка дисперсии.
#
# Именно это и должны показывать графики выше:
# - средние эмпирических кривых близки к теоретическим;
# - различия небольшие и объясняются конечным числом повторов (`1000`).

# %% [markdown]
# ## Пункт 2. Выборочные функции распределения и теоретические CDF
#
# Теперь по условию нужно взять уже не все `n`,
# а только два случая:
# - `n = 10`;
# - `n = 100`.
#
# Для этих размеров выборки сравним:
# - эмпирические функции распределения оценок;
# - теоретические функции распределения тех же оценок.
#
# Это позволяет увидеть уже не только среднее и дисперсию,
# но и форму распределения оценки целиком.

# %%
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

for column, n in enumerate((10, 100)):
    plot_ecdf_vs_theory(
        ax=axes[0, column],
        estimates=results_by_n[n]["mean_estimates"],
        cdf_function=theoretical_cdf_mean,
        n=n,
        title=f"Выборочное среднее, n = {n}",
    )
    plot_ecdf_vs_theory(
        ax=axes[1, column],
        estimates=results_by_n[n]["uncorrected_var_estimates"],
        cdf_function=theoretical_cdf_uncorrected_var,
        n=n,
        title=f"Неисправленная дисперсия, n = {n}",
    )
    plot_ecdf_vs_theory(
        ax=axes[2, column],
        estimates=results_by_n[n]["corrected_var_estimates"],
        cdf_function=theoretical_cdf_corrected_var,
        n=n,
        title=f"Исправленная дисперсия, n = {n}",
    )

fig.tight_layout()


# %% [markdown]
# Что важно увидеть на этих графиках:
# - для выборочного среднего ступенчатая эмпирическая CDF хорошо повторяет нормальную CDF;
# - для исправленной дисперсии основная масса значений сосредоточена около `1`;
# - для неисправленной дисперсии распределение заметно сдвинуто левее,
#   потому что её среднее меньше `1`.
#
# Кроме того, при `n = 100` совпадение эмпирической и теоретической CDF обычно лучше,
# чем при `n = 10`, потому что на больших выборках оценки становятся стабильнее.

# %% [markdown]
# ## Численные иллюстрации для `n = 10` и `n = 100`
#
# Чтобы выводы по графикам были совсем прозрачными,
# выведем для этих двух случаев эмпирические и теоретические средние и дисперсии.

# %%
distribution_rows = []

for n in (10, 100):
    mean_emp_mean, mean_emp_var = summarize_estimates(results_by_n[n]["mean_estimates"])
    unc_emp_mean, unc_emp_var = summarize_estimates(results_by_n[n]["uncorrected_var_estimates"])
    cor_emp_mean, cor_emp_var = summarize_estimates(results_by_n[n]["corrected_var_estimates"])

    mean_theory_mean, mean_theory_var = theoretical_mean_stats(n)
    unc_theory_mean, unc_theory_var = theoretical_uncorrected_var_stats(n)
    cor_theory_mean, cor_theory_var = theoretical_corrected_var_stats(n)

    distribution_rows.extend(
        [
            (
                f"c*, n={n}",
                f"{mean_emp_mean:.4f}",
                f"{mean_theory_mean:.4f}",
                f"{mean_emp_var:.4f}",
                f"{mean_theory_var:.4f}",
            ),
            (
                f"D_n*, n={n}",
                f"{unc_emp_mean:.4f}",
                f"{unc_theory_mean:.4f}",
                f"{unc_emp_var:.4f}",
                f"{unc_theory_var:.4f}",
            ),
            (
                f"S^2, n={n}",
                f"{cor_emp_mean:.4f}",
                f"{cor_theory_mean:.4f}",
                f"{cor_emp_var:.4f}",
                f"{cor_theory_var:.4f}",
            ),
        ]
    )

print_table(
    distribution_rows,
    headers=[
        "Оценка",
        "Эмп. среднее",
        "Теор. среднее",
        "Эмп. дисперсия",
        "Теор. дисперсия",
    ],
)


# %% [markdown]
# ## Проверка выполнения задания
#
# Здесь соберём короткий чек-лист и убедимся,
# что все необходимые пункты действительно выполнены.

# %%
for n in SAMPLE_SIZES:
    n = int(n)
    assert results_by_n[n]["mean_estimates"].shape == (N_REPEATS,)
    assert results_by_n[n]["uncorrected_var_estimates"].shape == (N_REPEATS,)
    assert results_by_n[n]["corrected_var_estimates"].shape == (N_REPEATS,)

print("Техническая проверка пройдена:")
print("- для каждого n получено ровно 1000 значений выборочного среднего;")
print("- для каждого n получено ровно 1000 значений неисправленной дисперсии;")
print("- для каждого n получено ровно 1000 значений исправленной дисперсии;")
print("- для n = 10 и n = 100 построены ECDF и наложены теоретические CDF.")


# %% [markdown]
# Проверим также содержательную часть задания:
#
# 1. **Сравнение средних и дисперсий оценок с теорией** выполнено,
#    потому что для всех `n` построены соответствующие графики.
# 2. **Проверка смещённости или несмещённости** выполнена,
#    потому что по графикам средних и по формулам видно:
#    - `c*` несмещённая;
#    - `D_n^*` смещённая;
#    - `S^2` несмещённая.
# 3. **Сравнение функций распределения** выполнено,
#    потому что для `n = 10` и `n = 100` построены ECDF и теоретические CDF.

# %% [markdown]
# ## Итоговые выводы
#
# 1. **Выборочное среднее `c* = \bar{x}`**
#
# Оно ведёт себя так, как и предсказывает теория:
# - эмпирическое среднее близко к `0`;
# - эмпирическая дисперсия близка к `1/n`;
# - эмпирическая функция распределения хорошо согласуется с нормальной.
#
# Значит, выборочное среднее является **несмещённой** оценкой математического ожидания.
#
# 2. **Неисправленная оценка дисперсии**
#
# У неё среднее близко не к `1`, а к `(n-1)/n`.
# Значит, она систематически занижает истинную дисперсию.
# Это и есть признак **смещённости**.
#
# 3. **Исправленная оценка дисперсии**
#
# У неё среднее близко к `1`, то есть к истинной дисперсии.
# Значит, эта оценка является **несмещённой**.
#
# 4. **Общий смысл моделирования**
#
# Массовое моделирование хорошо подтверждает теорию:
# когда мы много раз повторяем один и тот же эксперимент,
# свойства оценок становятся видны не только по формулам,
# но и по графикам, и по эмпирическим функциям распределения.
#
# Это и есть главный вывод работы:
# теоретические свойства оценок хорошо подтверждаются на практике,
# а различия между неисправленной и исправленной оценками дисперсии
# очень наглядно видны уже на сравнительно небольших выборках.
