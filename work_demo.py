import numpy as np
from scipy.optimize import minimize

# 定义机组的燃料消耗函数
def fuel_cost(M, a, b, c):
    return a * M ** 2 + b * M + c

# 定义总燃料消耗的目标函数
def total_fuel_cost(M):
    a = [0.0000451, 0.0000556, 0.0000527, 0.0000586]
    b = [0.2662, 0.2710, 0.2661, 0.2670]
    c = [14.062, 14.606, 13.481, 12.615]
    return sum(fuel_cost(M[i], a[i], b[i], c[i]) for i in range(4))

# 约束条件：总发电量以及每台机组的发电量范围
def power_constraint(M, P_total):
    return sum(M) - P_total

def solve_power_distribution(P_total):
    # 初始猜测
    initial_guess = [P_total / 4] * 4  # 将总发电量平均分配

    # 定义约束
    constraints = [{'type': 'eq', 'fun': power_constraint, 'args': [P_total]}]
    bounds = [(120, 300) for _ in range(4)]  # 每台机组的发电量在 120 MW 和 300 MW 之间

    # 求解最优化问题
    result = minimize(total_fuel_cost, initial_guess, bounds=bounds, constraints=constraints)

    if result.success:
        M_optimal = result.x
        total_cost = total_fuel_cost(M_optimal)
        print(f"总发电量 {P_total} MW 的最优分配: M1={M_optimal[0]:.2f}, M2={M_optimal[1]:.2f}, M3={M_optimal[2]:.2f}, M4={M_optimal[3]:.2f}")
        print(f"总耗煤量: {total_cost:.4f} g/kWh")
    else:
        print("优化失败")

# 测试
if __name__ == '__main__':
    test_min=600
    test_max=1000
    solve_power_distribution(test_min)
    solve_power_distribution(test_max)
