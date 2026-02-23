# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% id="AOsox_LCcdCn"
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from scipy.stats import moment
from scipy.optimize import minimize
import math


# %% [markdown] id="WtEyl-AVcyYy"
# ## 1. Метод моментов для равномерного распределения

# %% colab={"base_uri": "https://localhost:8080/", "height": 472} id="-CpttPUwS-Am" outputId="2f81f122-af15-4b9e-ea5d-a37fc6a79aed"
def method_of_moments_uniform(data):
    m1 = np.mean(data)
    m2 = np.mean(data ** 2)

    # Решение системы уравнений для нахождения a и b
    a_hat = m1 - np.sqrt(3 * (m2 - m1**2))
    b_hat = m1 + np.sqrt(3 * (m2 - m1**2))

    return a_hat, b_hat

# Генерация выборок для равномерного распределения
def generate_uniform_samples(n, a=0, b=1):
    return np.random.uniform(a, b, n)

# Функция для построения графика зависимости оценок от величины выборки
def plot_estimations_uniform():
    sample_sizes = range(10, 1000, 10)
    a_estimations = []
    b_estimations = []

    for n in sample_sizes:
        data = generate_uniform_samples(n)
        a_hat, b_hat = method_of_moments_uniform(data)
        a_estimations.append(a_hat)
        b_estimations.append(b_hat)

    plt.plot(sample_sizes, a_estimations, label="a_hat")
    plt.plot(sample_sizes, b_estimations, label="b_hat")
    plt.axhline(0, color='red', linestyle='--', label="True a=0")
    plt.axhline(1, color='green', linestyle='--', label="True b=1")
    plt.xlabel("Sample size")
    plt.ylabel("Estimates")
    plt.legend()
    plt.title("Method of Moments for Uniform Distribution")
    plt.show()

# Построение графика
plot_estimations_uniform()


# %% [markdown] id="Vdz-gFzxWeft"
# ## 2. Обобщенный метод моментов для равномерного распределения

# %% colab={"base_uri": "https://localhost:8080/", "height": 487} id="pzRC0BiPTiMW" outputId="0b8df916-3fcd-4723-c042-b584447c690f"
def theoretical_moments_uniform(a, b, num_moments):
    moments = []
    for k in range(1, num_moments + 1):
        moment_k = (b**k - a**k) / (k * (b - a))
        moments.append(moment_k)
    return moments

def generalized_moments_uniform(data, num_moments=3):
    """Обобщённый метод моментов для равномерного распределения"""
    empirical_moments = [np.mean(data ** k) for k in range(1, num_moments + 1)]

    if num_moments >= 2:
        m1 = empirical_moments[0]
        m2 = empirical_moments[1]

        a_hat = m1 - np.sqrt(3 * (m2 - m1**2))
        b_hat = m1 + np.sqrt(3 * (m2 - m1**2))

        return a_hat, b_hat
    else:
        raise ValueError("Number of moments must be at least 2")

def generate_uniform_samples(n, a=0, b=1):
    return np.random.uniform(a, b, n)

def plot_generalized_estimations_uniform(num_moments):
    sample_sizes = range(10, 1000, 10)
    a_estimations = []
    b_estimations = []

    for n in sample_sizes:
        data = generate_uniform_samples(n)
        a_hat, b_hat = generalized_moments_uniform(data, num_moments)
        a_estimations.append(a_hat)
        b_estimations.append(b_hat)

    plt.plot(sample_sizes, a_estimations, label="a_hat")
    plt.plot(sample_sizes, b_estimations, label="b_hat")
    plt.axhline(0, color='red', linestyle='--', label="True a=0")
    plt.axhline(1, color='green', linestyle='--', label="True b=1")
    plt.xlabel("Sample size")
    plt.ylabel("Estimates")
    plt.legend()
    plt.title(f"Generalized Method of Moments with {num_moments} moments")

plt.figure(figsize = (15, 5))
plt.subplot(1, 3, 1)
plot_generalized_estimations_uniform(num_moments=3)
plt.subplot(1, 3, 2)
plot_generalized_estimations_uniform(num_moments=5)
plt.subplot(1, 3, 3)
plot_generalized_estimations_uniform(num_moments=7)
plt.show()


# %% [markdown] id="uMLh9lNiXCsT"
# ## 3. Зависимость СКО от размера выборки

# %% colab={"base_uri": "https://localhost:8080/", "height": 472} id="5bLmm-sfWHAF" outputId="1a69dc26-a552-4d2b-eaea-e7edba0d01e7"
def theoretical_moments_uniform(a, b, num_moments):
    moments = []
    for k in range(1, num_moments + 1):
        moment_k = (b**k - a**k) / (k * (b - a))
        moments.append(moment_k)
    return moments

def generalized_moments_uniform(data, num_moments=3):
    """Обобщённый метод моментов для равномерного распределения"""
    empirical_moments = [np.mean(data ** k) for k in range(1, num_moments + 1)]

    if num_moments >= 2:
        m1 = empirical_moments[0]
        m2 = empirical_moments[1]

        a_hat = m1 - np.sqrt(3 * (m2 - m1**2))
        b_hat = m1 + np.sqrt(3 * (m2 - m1**2))

        return a_hat, b_hat
    else:
        raise ValueError("Number of moments must be at least 2")

def generate_uniform_samples(n, a=0, b=1):
    return np.random.uniform(a, b, n)

def compute_rmse_uniform(sample_size, num_samples=1000, num_moments=3):
    a_errors = []
    b_errors = []

    for _ in range(num_samples):
        data = generate_uniform_samples(sample_size)
        a_hat, b_hat = generalized_moments_uniform(data, num_moments)
        a_errors.append((a_hat - 0)**2)
        b_errors.append((b_hat - 1)**2)

    a_rmse = np.sqrt(np.mean(a_errors))
    b_rmse = np.sqrt(np.mean(b_errors))

    return a_rmse, b_rmse

def plot_rmse_uniform(num_moments_list):
    sample_sizes = range(10, 1000, 50)

    for num_moments in num_moments_list:
        a_rmses = []
        b_rmses = []
        for n in sample_sizes:
            a_rmse, b_rmse = compute_rmse_uniform(n, num_samples=1000, num_moments=num_moments)
            a_rmses.append(a_rmse)
            b_rmses.append(b_rmse)

        plt.plot(sample_sizes, a_rmses, label=f"a RMSE (moments={num_moments})")
        plt.plot(sample_sizes, b_rmses, label=f"b RMSE (moments={num_moments})")

    plt.xlabel("Sample size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("RMSE of Uniform Distribution Estimates (Generalized Method of Moments)")
    plt.show()

plot_rmse_uniform([3, 5, 7])


# %% [markdown] id="2fNDhdncXQVm"
# ## 4. Метод моментов для экспоненты

# %% id="d2g0Y31LX84h"
def method_of_moments(data):
    """Оценка параметра λ методом моментов."""
    mean_x = np.mean(data)
    return 1 / mean_x

def generalized_method_of_moments(data, moments):
    """Оценка параметра λ обобщенным методом моментов."""
    estimates = []
    for moment in moments:
        if moment == 1:
            estimates.append(1 / np.mean(data))
        elif moment == 2:
            estimates.append(np.sqrt(2 / np.mean(data**2)))
        elif moment == 3:
            estimates.append((6 / np.mean(data**3))**(1/3))
        elif moment == 4:
            estimates.append((24 / np.mean(data**4))**(1/4))
        elif moment == 5:
            estimates.append((120 / np.mean(data**5))**(1/5))
        elif moment == 6:
            estimates.append((720 / np.mean(data**6))**(1/6))
        elif moment == 7:
            estimates.append((5040 / np.mean(data**7))**(1/7))
    return estimates


# %% id="6S4n_z7yKN9-"
sample_sizes = range(10, 1000, 10)
std_devs = []
lambda_moments = []
lambda_gmm = []

for size in sample_sizes:
    sample = np.random.exponential(scale=1.0, size=size)
    std_devs.append(np.std(sample))

    lambda_moments.append(method_of_moments(sample))

    gmm_estimates = generalized_method_of_moments(sample, [1, 3, 5, 7])
    lambda_gmm.append(gmm_estimates)

# %% id="4o33O2laLYfz" outputId="d1ad1cc3-8ee2-45b1-b54a-02242790ef6f" colab={"base_uri": "https://localhost:8080/", "height": 956}
# Построение графиков
plt.figure(figsize=(15, 10))

# График СКО
plt.subplot(2, 2, 1)
plt.plot(sample_sizes, std_devs, marker='o')
plt.title('Зависимость СКО от размера выборки')
plt.xlabel('Размер выборки')
plt.ylabel('Стандартное отклонение (СКО)')
plt.grid()

# График оценок λ методом моментов
plt.subplot(2, 2, 2)
plt.plot(sample_sizes, lambda_moments, marker='o', label='Метод моментов')
plt.title('Оценка λ методом моментов')
plt.xlabel('Размер выборки')
plt.ylabel('Оценка λ')
plt.axhline(y=1, color='r', linestyle='--', label='Истинное значение λ=1')
plt.legend()
plt.grid()

# График оценок λ обобщенным методом моментов
plt.subplot(2, 2, 3)
for i in range(len(lambda_gmm[0])):  # Проходим по всем моментам
    plt.plot(sample_sizes, [l[i] for l in lambda_gmm], marker='o', label=f'Момент {i*2+1}')
plt.title('Оценка λ обобщенным методом моментов')
plt.xlabel('Размер выборки')
plt.ylabel('Оценка λ')
plt.axhline(y=1, color='r', linestyle='--', label='Истинное значение λ=1')
plt.legend()
plt.grid()

# Показать графики
plt.tight_layout()
plt.show()

# %% id="P-R9ONXKLaMY"
