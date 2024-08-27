import numpy as np

# 定义目标函数
# def objective_function(x):
#     return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1))

# # 定义等式约束
# def equality_constraints(x):
#     return np.array([
#               #sum(x) - 2,  # 示例等式约束1：sum(x) = n
#         #sum((i+1) * x[i] for i in range(2)) - (2 * (2 + 1)) // 2  # 示例等式约束2：sum((i+1)*x_i) = n(n+1)/2
#     ])  # 等式约束，m_1=2

# # 定义不等式约束
# def inequality_constraints(x):
#     return np.array([
#     ])  # 不等式约束，m_2=2

# # 定义盒式约束
# l_x = np.array([-2e+20,-2e+20])  # 下界
# u_x = np.array([2e+20,2e+20])  # 上界

# # 示例初始点
# p0 = np.array([-1.2,1])

def objective_function(x):
    return (x[0] - 2)**2 + (x[1] - 1)**2 + (x[2] - 3)**2  # 复杂目标函数

def equality_constraints(x):
    return np.array([
        x[0] + x[1] + x[2] - 6,  # 等式约束1
        x[0]**2 + x[1]**2 + x[2]**2 - 14  # 等式约束2
    ])  # 等式约束，m_1=2

def inequality_constraints(x):
    return np.array([
        x[0] - 3,  # 不等式约束1，x[0] >= 3
        0.5 - x[1]  # 不等式约束2，x[1] <= 4
    ])  # 不等式约束，m_2=2

# 定义盒式约束
l_x = np.array([-1.e+20,-1.e+20,-1.e+20])  # 下界
u_x = np.array([1.e+20, 1.e+20, 1.e+20])  # 上界

p0 = np.array([2, 2, 2])






def get_problem_data():
    return objective_function, equality_constraints, inequality_constraints, l_x, u_x, p0