import coptpy as cp
import numpy as np
import origin_solnp as solnp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

n = 10

# 定义盒式约束
l_x = np.full(n, -2)
u_x = np.full(n, 2)

 # 定义不等式约束
def inequality_constraints(x):
     return np.array([x[0] + x[1] - 1, 2 * x[0] - x[1] + 2])

# 添加松弛变量以将不等式约束转换为等式约束
def add_slack_variables(d_func, dL, dU, x0):
    d0 = d_func(x0)
    s0 = d0
    def eq_constraint(x_s):
        x = x_s[:len(x0)]
        s = x_s[len(x0):]
        return d_func(x) - s
    x_s0 = np.concatenate([x0, s0])
    cons = {'type': 'eq', 'fun': eq_constraint}
    return cons, x_s0

# 将第一个程序的功能集成到第二个程序
def solnp_solver_with_slack(obj_func, eq_constraints, ineq_constraints, l_x, u_x, p0, dL, dU, max_outer_iter=20, max_inner_iter=10, tol=1e-6):
    n = len(p0)
    m1 = len(eq_constraints(p0))
    rho = 1.0
    H = np.eye(n)
    y = np.zeros(m1)
    p_k = p0.copy()
    objective_values = []
    infeasibilities = []
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
        if v_k <= c_z * tol:
            rho = 0.0

        p_f_k = solnp.find_feasible_solution(p_k, eq_constraints, l_x, u_x)
        xi_k = p_f_k.copy()

        for i in range(max_inner_iter):
            xi_k_old = xi_k.copy()
            xi_k, lagrange_multiplier, success = solnp.inner_iteration(p_f_k, xi_k, p_k, y, rho, obj_func, eq_constraints, l_x, u_x, H, tol)
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

        if (abs(obj_func(xi_k) - obj_func(p_k)) / max(1, abs(obj_func(p_k))) <= epsilon_s and
            np.linalg.norm(solnp.project_to_box(xi_k - approx_fprime(xi_k, lambda x: solnp.augmented_lagrangian(x, y, rho, obj_func, eq_constraints), np.sqrt(np.finfo(float).eps)), l_x, u_x) - xi_k) > epsilon_a):
            print(f"Restarting at outer iteration {k+1}")
            H = np.diag(np.diag(H))
            continue

        if obj_func(xi_k) > obj_func(p_k) and v_k_new > v_k:
            print(f"Restarting due to increase in objective and infeasibility at outer iteration {k+1}")
            y = np.zeros(m1)
            H = np.diag(np.diag(H))
            continue

        print(f"Outer iteration {k+1}, optimal solution: {xi_k}, infeas: {solnp.infeas(xi_k, eq_constraints, l_x, u_x)}")

        if v_k_new < tol and abs(obj_func(xi_k) - obj_func(p_k)) / max(1, abs(obj_func(p_k))) < epsilon_s:
            print(f"Outer iteration {k+1} converged.")
            break

        p_k = xi_k.copy()

    return p_k, objective_values, infeasibilities, rhos

p0 = np.full(n, 1.2)
dL = np.array([0, 0])
dU = np.array([np.inf, np.inf])

solution, objective_values, infeasibilities, rhos = solnp_solver_with_slack(solnp.objective_function, solnp.equality_constraints, inequality_constraints, l_x, u_x, p0, dL, dU)

print("Final optimal solution:", solution)
print(objective_values)
print(infeasibilities)
print(rhos)

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
plt.savefig('nofilter.png')
