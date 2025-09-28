def simple_probability(m, n):
    return m / n

def logical_or(m, k, n):
    return (m + k) / n

def logical_and(m, k, n, l):
    p_a = m / n  # Вероятность первого события
    p_b = k / l  # Вероятность второго события
    return p_a * p_b

def expected_value(values, probabilities):
    result = 0.0
    for i in range(len(values)):
        result += values[i] * probabilities[i]
    return result

def conditional_probability(values):
    count_A1 = 0  # количество случаев, когда A=1
    count_A1_and_B1 = 0  # количество случаев, когда A=1 и B=1
    for a, b in values:
        if a == 1:
            count_A1 += 1
            if b == 1:
                count_A1_and_B1 += 1

    if count_A1 == 0:
        return 0.0  # Если нет случаев, когда A=1, возвращаем 0
    return count_A1_and_B1 / count_A1

def bayesian_probability(a: float, b: float, ba: float) -> float:
    return (ba * a) / b
