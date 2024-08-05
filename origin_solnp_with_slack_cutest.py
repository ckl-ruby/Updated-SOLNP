import numpy as np
from scipy.optimize import approx_fprime
import pycutest
import origin_solnp as solnp  # 请确保您有origin_solnp库
import matplotlib.pyplot as plt
import os
import pandas as pd

# 定义问题集
                #   'HS30', 'HS31', 'HS32', 'HS33', 'HS34', 'HS35', 'HS36', 'HS37', 'HS38', 'HS39', 'HS41', 'HS42', 'HS43', 'HS44', 'HS45', 'HS46', 'HS47', 'HS48',
                #   'HS49', 'HS50', 'HS51', 'HS52', 'HS53', 'HS55', 'HS56', 'HS59', 'HS60', 'HS61', 'HS62', 'HS63',
                #   'HS64', 'HS65', 'HS66', 'HS68', 'HS69', 'HS70', 'HS71', 'HS72', 'HS73', 'HS74', 'HS75', 'HS76',
                #   'HS77', 'HS78', 'HS79', 'HS80', 'HS81', 'HS83', 'HS84', 'HS85', 'HS86', 'HS87', 'HS88', 'HS89',
                #   'HS90', 'HS91', 'HS92', 'HS93', 'HS95', 'HS96', 'HS97', 'HS98', 'HS99', 'HS100', 'HS104', 'HS105',
                #   'HS109', 'HS110', 'HS114', 'HS118',
                #   'HS29', 'HS28', 'HS27', 'HS26', 'HS25', 'HS24', 'HS23', 'HS22', 
                #   'HS21', 'HS20', 'HS19', 'HS18', 'HS17', 'HS16', 'HS15', 'HS14', 
                #   'HS13', 'HS12', 'HS11', 'HS10'
problem_names = ['HS40']

# 创建保存结果的文件夹
output_folder = 'examples_cutest'
os.makedirs(output_folder, exist_ok=True)

# 定义目标函数
def objective_function(x, problem):
    return problem.obj(x)

# 定义等式约束
def equality_constraints(x, problem):
    if problem.m > 0:  # 检查是否有约束
        eq_constraints = []
        for i in range(problem.m):
            if problem.is_eq_cons[i]:  # 使用is_eq_cons来判断是否是等式约束
                ci = problem.cons(x, index=i)
                eq_constraints.append(ci)
        return np.array(eq_constraints)
    return np.array([])

# 定义不等式约束
def inequality_constraints(x, problem):
    if problem.m > 0:  # 检查是否有约束
        ineq_constraints = []
        for i in range(problem.m):
            if not problem.is_eq_cons[i]:  # 使用is_eq_cons来判断是否是不等式约束
                ci = problem.cons(x, index=i)
                ineq_constraints.append(ci)
        return np.array(ineq_constraints)
    return np.array([])

# 添加松弛变量以将不等式约束转换为等式约束
def add_slack_variables(d_func, dL, dU, x0):
    d0 = d_func(x0)
    s0 = d0 if d0.size > 0 else np.zeros_like(x0)
    def eq_constraint(x_s):
        x = x_s[:len(x0)]
        s = x_s[len(x0):]
        return d_func(x) - s
    x_s0 = np.concatenate([x0, s0])
    cons = {'type': 'eq', 'fun': eq_constraint}
    return cons, x_s0

# 记录评估次数的字典
evaluation_counts = {problem_name: 0 for problem_name in problem_names}

# 将第一个程序的功能集成到第二个程序
def solnp_solver_with_slack(obj_func, eq_constraints, ineq_constraints, l_x, u_x, p0, dL, dU, max_outer_iter=100, max_inner_iter=10, tol=1e-6, problem_name=None):
    global evaluation_counts  # 使用全局变量记录评估次数
    n = len(p0)
    eq_cons = eq_constraints(p0)
    m1 = len(eq_cons) if eq_cons is not None else 0
    rho = 1.0
    H = np.eye(n)
    y = np.zeros(m1)
    p_k = p0.copy()
    objective_values = []
    infeasibilities = []
    relative_diffs = []
    norm_projections = []
    solutions = []
    rhos = []
    objective_values.append(obj_func(p_k))
    infeasibilities.append(solnp.infeas(p_k, eq_constraints, l_x, u_x))
    rhos.append(rho)
    c_z = 1.2
    c_ir = 10.0
    c_rr = 5.0
    r_ir = 5.0
    r_rr = 0.2
    epsilon_s = 1e-4
    epsilon_a = 1e-2

    cons, x_s0 = add_slack_variables(ineq_constraints, dL, dU, p0)

    print(f"Initial solution: {p_k}, infeas: {solnp.infeas(p_k, eq_constraints, l_x, u_x)}")

    for k in range(max_outer_iter):
        v_k = solnp.infeas(p_k, eq_constraints, l_x, u_x)
        evaluation_counts[problem_name] += 1  # 记录外部迭代次数
        if v_k <= c_z * tol:
            rho = 0.0

        p_f_k = solnp.find_feasible_solution(p_k, eq_constraints, l_x, u_x)
        xi_k = p_f_k.copy()

        for i in range(max_inner_iter):
            xi_k_old = xi_k.copy()
            xi_k, lagrange_multiplier, success = solnp.inner_iteration(p_f_k, xi_k, p_k, y, rho, obj_func, eq_constraints, l_x, u_x, H, tol)
            evaluation_counts[problem_name] += 1  # 记录内部迭代次数
            if not success:
                print(f"Inner iteration {i+1} failed.")
                break

            sk = xi_k - xi_k_old
            t_k = approx_fprime(xi_k, lambda x: solnp.augmented_lagrangian(x, y, rho, obj_func, eq_constraints), np.sqrt(np.finfo(float).eps)) - approx_fprime(xi_k_old, lambda x: solnp.augmented_lagrangian(x, y, rho, obj_func, eq_constraints), np.sqrt(np.finfo(float).eps))

            if np.dot(sk, t_k) > 0:
                H = H + np.outer(t_k, t_k) / np.dot(t_k, sk) - np.dot(H, np.outer(sk, sk)).dot(H) / np.dot(sk, H.dot(sk))

            y = lagrange_multiplier

            xi_k = solnp.line_search(xi_k_old, xi_k, solnp.augmented_lagrangian, y, rho, obj_func, eq_constraints)

            print(f"Inner iteration {i+1}, xi_k: {xi_k}, infeasibility: {solnp.infeas(xi_k, eq_constraints, l_x, u_x)}, lagrange multipliers: {lagrange_multiplier}")
            print(f"Hessian matrix H:\n{H}")

            if np.linalg.norm(sk) < tol:
                print(f"Inner iteration {i+1} converged.")
                break

        gk = eq_constraints(xi_k)
        y = y - rho * gk

        v_k_new = solnp.infeas(xi_k, eq_constraints, l_x, u_x)
        if v_k_new >= c_ir * v_k:
            rho = r_ir * rho
        elif v_k_new <= c_rr * v_k:
            rho = r_rr * rho

        objective_values.append(obj_func(xi_k))
        infeasibilities.append(v_k_new)
        solutions.append(xi_k.copy())
        rhos.append(rho)

        # 计算相对差值和盒式约束投影的范数
        relative_diff = abs(obj_func(xi_k) - obj_func(p_k)) / max(1, abs(obj_func(p_k)))
        norm_projection = np.linalg.norm(solnp.project_to_box(xi_k - approx_fprime(xi_k, lambda x: solnp.augmented_lagrangian(x, y, rho, obj_func, eq_constraints), np.sqrt(np.finfo(float).eps)), l_x, u_x) - xi_k)
        
        relative_diffs.append(relative_diff)
        norm_projections.append(norm_projection)

        if (relative_diff <= epsilon_s and norm_projection > epsilon_a):
            print(f"Restarting at outer iteration {k+1}")
            H = np.diag(np.diag(H))
            continue

        if obj_func(xi_k) > obj_func(p_k) and v_k_new > v_k:
            print(f"Restarting due to increase in objective and infeasibility at outer iteration {k+1}")
            y = np.zeros(m1)
            H = np.diag(np.diag(H))
            continue

        print(f"Outer iteration {k+1}, optimal solution: {xi_k}, infeas: {solnp.infeas(xi_k, eq_constraints, l_x, u_x)}")

        if v_k_new < tol and relative_diff < epsilon_s:
            print(f"Outer iteration {k+1} converged.")
            break

        p_k = xi_k.copy()

    return p_k, objective_values, infeasibilities, rhos, relative_diffs, norm_projections

for problem_name in problem_names:
    problem = pycutest.import_problem(problem_name)
    
    l_x = problem.bl
    u_x = problem.bu
    p0 = problem.x0
    dL = np.array([])
    dU = np.array([])

    solution, objective_values, infeasibilities, rhos, relative_diffs, norm_projections = solnp_solver_with_slack(
        lambda x: objective_function(x, problem),
        lambda x: equality_constraints(x, problem),
        lambda x: inequality_constraints(x, problem),
        l_x, u_x, p0, dL, dU, problem_name=problem_name
    )

    print(f"Final optimal solution for {problem_name}:", solution)
    print(objective_values)
    print(infeasibilities)
    print(rhos)

    # 可视化
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.plot(range(len(objective_values)), objective_values, label='Objective Function Value')
    plt.xlabel('Outer Iteration')
    plt.ylabel('Objective Function Value')
    plt.title(f'Objective Function Value over Iterations for {problem_name}')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(range(len(infeasibilities)), infeasibilities, label='Infeasibility')
    plt.xlabel('Outer Iteration')
    plt.ylabel('Infeasibility')
    plt.title(f'Infeasibility over Iterations for {problem_name}')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(range(len(relative_diffs)), relative_diffs, label='Relative Difference')
    plt.xlabel('Outer Iteration')
    plt.ylabel('Relative Difference')
    plt.title(f'Relative Difference over Iterations for {problem_name}')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(range(len(norm_projections)), norm_projections, label='Norm of Projection')
    plt.xlabel('Outer Iteration')
    plt.ylabel('Norm of Projection')
    plt.title(f'Norm of Projection over Iterations for {problem_name}')
    plt.legend()

    plt.suptitle(f'Optimization Results for {problem_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{problem_name}_results.png'))
    plt.close()

# 将评估次数保存为CSV文件并输出为表格
evaluation_df = pd.DataFrame(list(evaluation_counts.items()), columns=['Problem', 'Evaluation Count'])
evaluation_df.to_csv(os.path.join(output_folder, 'evaluation_counts_{max_outer_iter}.csv'), index=False)
print(evaluation_df)
