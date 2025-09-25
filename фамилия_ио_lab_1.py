# -*- coding: utf-8 -*-
"""
Лабораторная работа 1 — функции по теории вероятностей.

Файл подготовлен: Гертер С.Е.
Имя файла: Гертер_СЕ_lab_1.py

Файл содержит только определения функций и не выполняет код при импорте.
"""


def simple_probability(m, n):
    """
    Простая вероятность: m успешных исходов из n.
    Возвращает float. Если n == 0, возвращает 0.0.
    """
    if n == 0:
        return 0.0
    return m / n


def logical_or(m, k, n):
    """
    Вероятность двух несовместных исходов: P(A or B) = P(A) + P(B).
    m — число успешных исходов A, k — число успешных исходов B, n — общее число исходов.
    Если n == 0, возвращает 0.0.
    """
    if n == 0:
        return 0.0
    return (m + k) / n


def logical_and(m, k, n, l):
    """
    Вероятность двух совместных (независимых) исходов: P(A and B) = P(A) * P(B).
    m/n — вероятность A, k/l — вероятность B.
    Если n == 0 или l == 0, возвращает 0.0.
    """
    if n == 0 or l == 0:
        return 0.0
    return (m / n) * (k / l)


def expected_value(values, probabilities):
    """
    Математическое ожидание дискретной величины.
    values — кортеж значений, probabilities — кортеж соответствующих вероятностей.
    В случае разных длин берётся минимальная длина (zip).
    """
    total = 0.0
    for v, p in zip(values, probabilities):
        total += v * p
    return total


def conditional_probability(values):
    """
    Условная вероятность P(B=1 | A=1).
    values — кортеж пар (a, b), где a и b ∈ {0,1}.
    Если нет случаев A=1, возвращается 0.0.
    """
    count_A1 = 0
    count_A1_B1 = 0
    for pair in values:
        a = pair[0]
        b = pair[1]
        if a == 1:
            count_A1 += 1
            if b == 1:
                count_A1_B1 += 1
    if count_A1 == 0:
        return 0.0
    return count_A1_B1 / count_A1


def bayesian_probability(a, b, ba):
    """
    Теорема Байеса: P(A|B) = P(A) * P(B|A) / P(B).
    a = P(A), b = P(B), ba = P(B|A).
    Если b == 0, возвращается 0.0.
    """
    if b == 0:
        return 0.0
    return (a * ba) / b
