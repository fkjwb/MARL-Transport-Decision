import random


def generate_arrays(T, a_range, b_range, sum_range):
    """
    生成两个长度为 T 的数组 A 和 B，
    满足对每个 i：
        sum_range[0] <= A[i] + B[i] <= sum_range[1]
    且 A[i], B[i] 都在各自范围内，并且 >= 0

    参数：
        T: 数组长度
        a_range: A 中元素范围，格式 (a_min, a_max)
        b_range: B 中元素范围，格式 (b_min, b_max)
        sum_range: 每对元素和的范围，格式 (sum_min, sum_max)

    返回：
        A, B
    """
    a_min, a_max = a_range
    b_min, b_max = b_range
    sum_min, sum_max = sum_range

    if a_min < 0 or b_min < 0:
        raise ValueError("元素下限必须 >= 0")

    if a_min > a_max or b_min > b_max:
        raise ValueError("元素范围无效")

    if sum_min > sum_max:
        raise ValueError("和的范围无效")

    A = []
    B = []

    for _ in range(T):
        valid_pairs = []

        for a in range(a_min, a_max + 1):
            for b in range(b_min, b_max + 1):
                s = a + b
                if sum_min <= s <= sum_max:
                    valid_pairs.append((a, b))

        if not valid_pairs:
            raise ValueError("当前参数下无可行解，请调整元素范围或和的范围")

        a, b = random.choice(valid_pairs)
        A.append(a)
        B.append(b)

    return A, B


if __name__ == "__main__":
    # ===== 参数设置 =====
    T = 8
    a_range = (0, 3)      # A数组元素范围
    b_range = (0, 3)      # B数组元素范围
    sum_range = (0, 3)    # 每对元素之和范围，例如 A[i] + B[i] <= 40

    A, B = generate_arrays(T, a_range, b_range, sum_range)

    print("A =", A)
    print("B =", B)
    print("\n逐项检查：")
    for i, (a, b) in enumerate(zip(A, B), start=1):
        print(f"A{i}={a}, B{i}={b}, A{i}+B{i}={a+b}")