import coptpy as cp
import numpy as np
from scipy.optimize import approx_fprime, minimize_scalar
from coptpy import COPT
import matplotlib.pyplot as plt

n = 100
def objective_function(x):
    return sum(100 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1))

# 定义等式约束 (m_1个约束，n维)
def equality_constraints(x):
    n = len(x)
    return np.array([
        sum(x) - n,  # 示例等式约束1：sum(x) = n
        sum((i + 1) * x[i] for i in range(n)) - (n * (n + 1)) // 2  # 示例等式约束2：sum((i+1)*x_i) = n(n+1)/2
    ])

# 定义不等式约束 (m_2个约束，n维)
def inequality_constraints(x):
    return np.array([
        x[0] - 1,  # 示例不等式约束1：x[0] >= 1
        2 - x[1]   # 示例不等式约束2：x[1] <= 2
    ])

# 定义盒式约束
l_x = np.full(n, -2)  # 每个变量的下界
u_x = np.full(n, 2)   # 每个变量的上界

# 定义松弛变量的初值
def initialize_slack_variables(ineq_constraints, x0):
    return ineq_constraints(x0)

# 将不等式约束转换为等式约束
def convert_ineq_to_eq_constraints(x, s):
    ineqs = inequality_constraints(x)
    return ineqs - s

# 结合等式约束和转换后的不等式约束
def combined_constraints(x, s):
    eq_constraints = equality_constraints(x)
    converted_ineq_constraints = convert_ineq_to_eq_constraints(x, s)
    return np.concatenate((eq_constraints, converted_ineq_constraints))

# 定义增广拉格朗日函数
def augmented_lagrangian(x, s, y, rho, obj_func, eq_constraints):
    lagr = obj_func(x)
    g = eq_constraints(x, s)
    lagr -= np.dot(y, g)
    lagr += (rho / 2) * np.linalg.norm(g)**2
    return lagr

# 定义 infeas 函数
def infeas(x, s, eq_constraints, l_x, u_x):
    g = eq_constraints(x, s)
    infeas_eq = np.sum(g**2)
    infeas_bounds = np.sum(np.maximum(0, x - u_x)**2) + np.sum(np.maximum(0, l_x - x)**2)
    return np.sqrt(infeas_eq + infeas_bounds)

# 计算约束的雅可比矩阵
def jacobian_eq_constraints(x, s, eq_constraints):
    epsilon = np.sqrt(np.finfo(float).eps)
    m1 = len(eq_constraints(x, s))
    n = len(x)
    jacobian = np.zeros((m1, n))
    for i in range(m1):
        jacobian[i, :] = approx_fprime(x, lambda x: eq_constraints(x, s)[i], epsilon)
    return jacobian

# 找到一个内部（或近乎可行）的解
def find_feasible_solution(p_k, s_k, eq_constraints, l_x, u_x):
    J = jacobian_eq_constraints(p_k, s_k, eq_constraints)
    gk = eq_constraints(p_k, s_k)
    if np.all(np.isinf(l_x)) and np.all(np.isinf(u_x)):  # 无盒式约束的情况
        p_f_k = p_k - np.linalg.pinv(J) @ gk
    else:  # 有盒式约束的情况
        env = cp.Envr()
        model = env.createModel("feasible_solution")

        # 定义变量
        n = len(p_k)
        x = [model.addVar(lb=l_x[i], ub=u_x[i]) for i in range(n)]
        tau = model.addVar(lb=0, ub=cp.COPT.INFINITY)  # 辅助变量 tau

        # 定义目标函数：最小化 tau
        model.setObjective(tau, COPT.MINIMIZE)

        # 定义线性约束 J^k * (x - p_k) - g(x^k) * tau = -g(x^k)
        for i in range(len(gk)):
            model.addConstr(cp.quicksum(J[i, j] * (x[j] - p_k[j]) for j in range(n)) - gk[i] * tau == -gk[i])

        # 设置参数
        model.setParam(COPT.Param.TimeLimit, 60)
        model.setParam(COPT.Param.FeasTol, 1e-9)
        model.setParam(COPT.Param.Logging, 0)  # 关闭COPT日志输出

        # 求解问题
        model.solve()

        if model.status == COPT.OPTIMAL:
            p_f_k = np.array([x[i].x for i in range(n)])
        else:
            p_f_k = p_k.copy()  # 如果求解失败，则保持原点
    return p_f_k

# 定义 QP 子问题的目标函数并使用 COPT 求解器求解
def solve_qp_subproblem_with_copt(xi_k, s_k, H, gradient_L, J, p_f_k, p_k, l_x, u_x, tol):
    env = cp.Envr()
    model = env.createModel("qp_subproblem")

    # 定义变量
    n = len(xi_k)
    x = [model.addVar(lb=l_x[i], ub=u_x[i]) for i in range(n)]

    # 定义目标函数 0.5 * dx.T * H * dx + gradient_L.T * dx
    obj = 0.5 * cp.quicksum(H[i, j] * (x[i] - xi_k[i]) * (x[j] - xi_k[j]) for i in range(n) for j in range(n))
    obj += cp.quicksum(gradient_L[i] * (x[i] - xi_k[i]) for i in range(n))
    model.setObjective(obj, COPT.MINIMIZE)

    # 定义线性约束 J * (x - p_k) = J * (p_f_k - p_k)
    constraint_rhs = J.dot(p_f_k - p_k)
    for i in range(len(constraint_rhs)):
        model.addConstr(cp.quicksum(J[i, j] * (x[j] - p_k[j]) for j in range(n)) == constraint_rhs[i])

    # 设置参数
    model.setParam(COPT.Param.TimeLimit, 60)
    model.setParam(COPT.Param.FeasTol, tol)
    model.setParam(COPT.Param.Logging, 0)  # 关闭COPT日志输出

    # 求解问题
    model.solve()

    if model.status == COPT.OPTIMAL:
        x_opt = np.array([x[i].x for i in range(n)])
        return x_opt, True
    else:
        return xi_k, False

# 定义滤波器方法检查函数
def filter_method_check(fk, hk, filter_set):
    for (fi, hi) in filter_set:
        if fk >= fi and hk >= hi:
            return False
    return True

def line_search(xi_k_old, xi_k, s_k, augmented_lagrangian, y, rho, obj_func, eq_constraints):
    def F(alpha):
        return augmented_lagrangian(alpha * xi_k_old + (1 - alpha) * xi_k, s_k, y, rho, obj_func, eq_constraints)
    
    result = minimize_scalar(F, bounds=(0, 1), method='bounded')
    alpha_star = result.x
    return alpha_star * xi_k_old + (1 - alpha_star) * xi_k

def inner_iteration(p_f_k, s_k, xi_k, p_k, yk, rho, obj_func, eq_constraints, l_x, u_x, H, tol):
    n = len(p_f_k)
    m1 = len(eq_constraints(p_f_k, s_k))
    
    J = jacobian_eq_constraints(p_k, s_k, eq_constraints)  # 这里使用的是 p_k 而不是 xi_k
    gk = eq_constraints(p_k, s_k)

    # 计算增广拉格朗日函数的梯度
    gradient_L = approx_fprime(xi_k, lambda x: augmented_lagrangian(x, s_k, yk, rho, obj_func, eq_constraints), np.sqrt(np.finfo(float).eps))
    
    # 使用 COPT 求解 QP 子问题
    xi_k, success = solve_qp_subproblem_with_copt(xi_k, s_k, H, gradient_L, J, p_f_k, p_k, l_x, u_x, tol)

    if success:
        # 计算拉格朗日乘子，使用伪逆
        lambda_k = -np.linalg.pinv(J @ J.T) @ (J @ (xi_k - p_k) + gk)
        return xi_k, lambda_k, success  # 返回解和拉格朗日乘子
    else:
        return xi_k, np.zeros(m1), success

def project_to_box(x, l_x, u_x):
    return np.minimum(np.maximum(x, l_x), u_x)

def solnp_solver(obj_func, eq_constraints, ineq_constraints, l_x, u_x, p0, max_outer_iter=100, max_inner_iter=10, tol=1e-6):
    n = len(p0)  # 变量维数
    m1 = len(eq_constraints(p0))  # 等式约束的个数
    m2 = len(ineq_constraints(p0))  # 不等式约束的个数
    rho = 1.0
    H = np.eye(n)  # 初始化Hessian矩阵
    y = np.zeros(m1 + m2)  # 初始化拉格朗日乘子
    p_k = p0.copy()
    s_k = initialize_slack_variables(ineq_constraints, p0)  # 初始化松弛变量
    
    filter_set = set()
    
    c_z = 1.2  # c_z 的值需要根据具体问题调整
    c_ir = 10.0  # c_ir 的值需要根据具体问题调整
    c_rr = 5.0  # c_rr 的值需要根据具体问题调整
    r_ir = 5.0  # r_ir 的值需要根据具体问题调整
    r_rr = 0.2  # r_rr 的值需要根据具体问题调整
    epsilon_s = 1e-4  # 停止准则的相对差值
    epsilon_a = 1e-2  # 用于检查重启条件的较大常数

    # 定义用于绘图的变量
    objective_values = []
    infeasibilities = []
    solutions = []
    objective_values.append(objective_function(p_k))
    infeasibilities.append(infeas(p_k, s_k, combined_constraints, l_x, u_x))
    
    print(f"Initial solution: {p_k}, infeas: {infeas(p_k, s_k, combined_constraints, l_x, u_x)}")

    for k in range(max_outer_iter):
        v_k = infeas(p_k, s_k, combined_constraints, l_x, u_x)
        if v_k <= c_z * tol:
            rho = 0.0

        # 寻找内部（或近乎可行）的解
        p_f_k = find_feasible_solution(p_k, s_k, combined_constraints, l_x, u_x)
        xi_k = p_f_k.copy()  # 初始化内迭代起点

        for i in range(max_inner_iter):
            xi_k_old = xi_k.copy()  # 存储上一次内迭代的点

            # 内迭代：解决线性化后的二次规划问题
            xi_k, lagrange_multiplier, success = inner_iteration(
                p_f_k, s_k, xi_k, p_k, y, rho, obj_func, combined_constraints, l_x, u_x, H, tol)

            if not success:
                print(f"Inner iteration {i + 1} failed.")
                break

            # 更新Hessian矩阵
            sk = xi_k - xi_k_old  # 使用相邻两个内迭代点
            t_k = approx_fprime(xi_k, lambda x: augmented_lagrangian(x, s_k, y, rho, obj_func, combined_constraints), np.sqrt(np.finfo(float).eps)) - approx_fprime(xi_k_old, lambda x: augmented_lagrangian(x, s_k, y, rho, obj_func, combined_constraints), np.sqrt(np.finfo(float).eps))
            
            if np.dot(sk, t_k) > 0:
                H = H + np.outer(t_k, t_k) / np.dot(t_k, sk) - np.dot(H, np.outer(sk, sk)).dot(H) / np.dot(sk, H.dot(sk))

            y = lagrange_multiplier  # 使用二次规划子问题的拉格朗日乘子作为对偶变量

            # 在这里进行线搜索
            xi_k = line_search(xi_k_old, xi_k, s_k, augmented_lagrangian, y, rho, obj_func, combined_constraints)

            # 计算目标函数和约束违反值
            fk = objective_function(xi_k)
            hk = infeas(xi_k, s_k, combined_constraints, l_x, u_x)

            # 滤波器方法检查
            if filter_method_check(fk, hk, filter_set):
                filter_set.add((fk, hk))

                print(f"Inner iteration {i + 1}, xi_k: {xi_k}, infeasibility: {hk}, lagrange multipliers: {lagrange_multiplier}")
                print(f"Hessian matrix H:\n{H}")


                # 检查内迭代收敛条件
                if np.linalg.norm(sk) < tol:
                    print(f"Inner iteration {i + 1} converged.")
                    break
            else:
                # 回溯线搜索
                alpha = 1.0
                while alpha > 1e-4:
                    alpha *= 0.5
                    xi_k_temp = xi_k_old + alpha * (xi_k - xi_k_old)
                    fk_temp = objective_function(xi_k_temp)
                    hk_temp = infeas(xi_k_temp, s_k, combined_constraints, l_x, u_x)
                    if filter_method_check(fk_temp, hk_temp, filter_set):
                        xi_k = xi_k_temp
                        fk = fk_temp
                        hk = hk_temp
                        break
                else:
                    print(f"Inner iteration {i + 1} failed due to infeasible point.")
                    break

        # 更新外迭代参数
        gk = combined_constraints(xi_k, s_k)
        y = y - rho * gk  # 更新拉格朗日乘子

        v_k_new = infeas(xi_k, s_k, combined_constraints, l_x, u_x)
        if v_k_new >= c_ir * v_k:
            rho = r_ir * rho
        elif v_k_new <= c_rr * v_k:
            rho = r_rr * rho

        # 保存当前迭代的目标函数值和约束违反值
        objective_values.append(objective_function(xi_k))
        infeasibilities.append(v_k_new)
        solutions.append(xi_k.copy())

        # 检查是否需要重启
        if (abs(objective_function(xi_k) - objective_function(p_k)) / max(1, abs(objective_function(p_k))) <= epsilon_s and
            np.linalg.norm(project_to_box(xi_k - approx_fprime(xi_k, lambda x: augmented_lagrangian(x, s_k, y, rho, obj_func, combined_constraints), np.sqrt(np.finfo(float).eps)), l_x, u_x) - xi_k) > epsilon_a):
            print(f"Restarting at outer iteration {k + 1}")
            H = np.diag(np.diag(H))  # 重置 Hessian 矩阵，只保留对角线元素
            continue

        if objective_function(xi_k) > objective_function(p_k) and v_k_new > v_k:
            print(f"Restarting due to increase in objective and infeasibility at outer iteration {k + 1}")
            y = np.zeros(m1 + m2)  # 将拉格朗日乘子设为 0
            H = np.diag(np.diag(H))  # 重置 Hessian 矩阵，只保留对角线元素
            continue

        # 打印当前最优解和 infeas 值
        print(f"Outer iteration {k + 1}, optimal solution: {xi_k}, infeas: {infeas(xi_k, s_k, combined_constraints, l_x, u_x)}")

        # 检查外迭代收敛条件
        if v_k_new < tol and abs(objective_function(xi_k) - objective_function(p_k)) / max(1, abs(objective_function(p_k))) < epsilon_s:
            print(f"Outer iteration {k + 1} converged.")
            break

        # 更新 p_k
        p_k = xi_k.copy()

    return p_k, objective_values, infeasibilities

p0 = np.full(n, 1.2)
# 运行求解器
solution, objective_values, infeasibilities = solnp_solver(objective_function, equality_constraints, inequality_constraints, l_x, u_x, p0)
print("Final optimal solution:", solution)
print(objective_values)
print(infeasibilities)

# 绘制目标函数值和约束违反值的变化并保存图像
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(objective_values)), objective_values, label='Objective Function Value')
plt.xlabel('Outer Iteration')
plt.ylabel('Objective Function Value')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(len(infeasibilities)), infeasibilities, label='Infeasibility')
plt.xlabel('Outer Iteration')
plt.ylabel('Infeasibility')
plt.legend()

plt.tight_layout()
plt.savefig('filter.png')
