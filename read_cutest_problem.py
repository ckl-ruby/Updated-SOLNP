import pycutest
import numpy as np 
import matplotlib.pyplot as plt

# 从PyCUTEst读取问题
problem_name = 'HS12'  # 替换为你想要解决的问题名称
problem = pycutest.import_problem(problem_name)

# 定义目标函数
def objective_function(x):
    return problem.obj(x)

# 定义等式约束
def equality_constraints(x):
    if problem.m > 0:  # 检查是否有约束
        eq_constraints = []
        for i in range(problem.m):
            if problem.is_eq_cons[i]:  # 使用is_eq_cons来判断是否是等式约束
                ci = problem.cons(x, index=i)
                eq_constraints.append(ci)
        return np.array(eq_constraints)
    return np.array([])

# 定义不等式约束 (转换为 g(x) >= 0 的形式)
def inequality_constraints(x):
    if problem.m > 0:  # 检查是否有约束
        ineq_constraints = []
        for i in range(problem.m):
            if not problem.is_eq_cons[i]:  # 仅处理不等式约束
                ci = problem.cons(x, index=i)
                if problem.cl[i] > -np.inf:  # cl <= ci(x)
                    ineq_constraints.append(ci - problem.cl[i])  # 转换为 ci(x) - cl >= 0
                if problem.cu[i] < np.inf:  # ci(x) <= cu
                    ineq_constraints.append(problem.cu[i] - ci)  # 转换为 cu - ci(x) >= 0
        return np.array(ineq_constraints)
    return np.array([])

# 定义变量的上下界
l_x = problem.bl if problem.bl is not None else -np.inf * np.ones(problem.n)
u_x = problem.bu if problem.bu is not None else np.inf * np.ones(problem.n)

# 设定初始解
p0 = problem.x0

# 返回定义的函数和参数
def get_problem_data():
    return objective_function, equality_constraints, inequality_constraints, l_x, u_x, p0

if __name__ == "__main__":
    # 打印目标函数
    print("Objective function at p0:", objective_function(p0))
    
    # 打印等式约束
    eq_constraints = equality_constraints(p0)
    print("Equality constraints at p0:", eq_constraints)
   
    
    # 打印不等式约束
    ineq_constraints = inequality_constraints(p0)
    print("Inequality constraints at p0:", ineq_constraints)
   
    
    # 打印变量的上下界
    print("Variable lower bounds:", l_x)
    print("Variable upper bounds:", u_x)
    
    # 打印初始解
    print("Initial point p0:", p0)
