# Gerter_SE_lab_1.py

def simple_probability(m: int, n: int) -> float:
    """
    Вычисляет простую вероятность: P = m / n
    """
    if n == 0:
        return 0.0
    return m / n


def logical_or(m: int, k: int, n: int) -> float:
    """
    Вероятность двух несовместных событий:
    P(A ∪ B) = (m + k) / n
    """
    if n == 0:
        return 0.0
    return (m + k) / n


def logical_and(m: int, k: int, n: int, l: int) -> float:
    """
    Вероятность двух совместных событий:
    P(A ∩ B) = (m / n) * (k / l)
    """
    if n == 0 or l == 0:
        return 0.0
    return (m / n) * (k / l)


def expected_value(values: tuple, probabilities: tuple) -> float:
    """
    Математическое ожидание дискретной величины:
    E[X] = Σ (x_i * p_i)
    """
    if len(values) != len(probabilities):
        raise ValueError("Длины кортежей должны совпадать")
    return sum(v * p for v, p in zip(values, probabilities))


def conditional_probability(values: tuple) -> float:
    """
    Условная вероятность:
    P(B=1 | A=1) = P(A=1 и B=1) / P(A=1)
    values – кортеж пар (a, b), где a и b ∈ {0,1}
    """
    count_A1 = 0
    count_A1B1 = 0
    for a, b in values:
        if a == 1:
            count_A1 += 1
            if b == 1:
                count_A1B1 += 1
    if count_A1 == 0:
        return 0.0
    return count_A1B1 / count_A1


def bayesian_probability(a: float, b: float, ba: float) -> float:
    """
    Теорема Байеса:
    P(A|B) = P(B|A) * P(A) / P(B)
    """
    if b == 0:
        return 0.0
    return (ba * a) / b
