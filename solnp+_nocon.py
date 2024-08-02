import numpy as np
from scipy.optimize import approx_fprime, minimize_scalar
import matplotlib.pyplot as plt

n = 100

def objective_function(x):
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1))

def augmented_lagrangian(x, y, rho, obj_func):
    return obj_func(x)

def infeas(x, eq_constraints=None, l_x=None, u_x=None):
    return 0

def jacobian_eq_constraints(x, eq_constraints=None):
    return np.zeros((0, len(x)))

def find_feasible_solution(p_k, eq_constraints=None, l_x=None, u_x=None):
    return p_k.copy()

def solve_qp_subproblem_with_copt(xi_k, H, gradient_L, J, p_f_k, p_k, l_x=None, u_x=None, tol=1e-6):
    n = len(xi_k)
    x_opt = np.linalg.solve(H, -gradient_L)
    return xi_k + x_opt, True

def line_search(xi_k_old, xi_k, augmented_lagrangian, y, rho, obj_func):
    def F(alpha):
        return augmented_lagrangian(alpha * xi_k_old + (1 - alpha) * xi_k, y, rho, obj_func)
    
    result = minimize_scalar(F, bounds=(0, 1), method='bounded')
    alpha_star = result.x
    return alpha_star * xi_k_old + (1 - alpha_star) * xi_k

def inner_iteration(p_f_k, xi_k, p_k, yk, rho, obj_func, eq_constraints=None, l_x=None, u_x=None, H=np.eye(n), tol=1e-6):
    n = len(p_f_k)
    J = jacobian_eq_constraints(xi_k)
    gradient_L = approx_fprime(xi_k, lambda x: augmented_lagrangian(x, yk, rho, obj_func), np.sqrt(np.finfo(float).eps))
    xi_k, success = solve_qp_subproblem_with_copt(xi_k, H, gradient_L, J, p_f_k, p_k, l_x, u_x, tol)

    if success:
        lambda_k = np.zeros(len(J))
        return xi_k, lambda_k, success
    else:
        return xi_k, np.zeros(len(J)), success

def project_to_box(x, l_x=None, u_x=None):
    return x

def solnp_solver(obj_func, eq_constraints=None, l_x=None, u_x=None, p0=np.ones(n), max_outer_iter=100, max_inner_iter=10, tol=1e-6):
    n = len(p0)
    rho = 1.0
    H = np.eye(n)
    y = np.zeros(0)
    p_k = p0.copy()
    
    objective_values = [objective_function(p_k)]
    infeasibilities = [infeas(p_k)]
    solutions = []
    rhos = [rho]
    relative_diff_values = [0]
    box_project_norms = [0]
    
    c_z = 1.2
    c_ir = 10.0
    c_rr = 5.0
    r_ir = 5.0
    r_rr = 0.2
    epsilon_s = 1e-4
    epsilon_a = 1e-2

    print(f"Initial solution: {p_k}")

    for k in range(max_outer_iter):
        v_k = infeas(p_k)
        if v_k <= c_z * tol:
            rho = 0.0

        p_f_k = find_feasible_solution(p_k)
        xi_k = p_f_k.copy()

        for i in range(max_inner_iter):
            xi_k_old = xi_k.copy()
            xi_k, lagrange_multiplier, success = inner_iteration(
                p_f_k, xi_k, p_k, y, rho, obj_func, eq_constraints, l_x, u_x, H, tol)

            if not success:
                print(f"Inner iteration {i+1} failed.")
                break

            sk = xi_k - xi_k_old
            t_k = approx_fprime(xi_k, lambda x: augmented_lagrangian(x, y, rho, obj_func), np.sqrt(np.finfo(float).eps)) - approx_fprime(xi_k_old, lambda x: augmented_lagrangian(x, y, rho, obj_func), np.sqrt(np.finfo(float).eps))
            
            if np.dot(sk, t_k) > 0:
                H = H + np.outer(t_k, t_k) / np.dot(t_k, sk) - np.dot(H, np.outer(sk, sk)).dot(H) / np.dot(sk, H.dot(sk))

            y = lagrange_multiplier
            xi_k = line_search(xi_k_old, xi_k, augmented_lagrangian, y, rho, obj_func)

            print(f"Inner iteration {i+1}, xi_k: {xi_k}, lagrange multipliers: {lagrange_multiplier}")
            print(f"Hessian matrix H:\n{H}")

            if np.linalg.norm(sk) < tol:
                print(f"Inner iteration {i+1} converged.")
                break

        gk = np.zeros(0)
        y = y - rho * gk

        v_k_new = infeas(xi_k)
        if v_k_new >= c_ir * v_k:
            rho = r_ir * rho
        elif v_k_new <= c_rr * v_k:
            rho = r_rr * rho

        objective_values.append(objective_function(xi_k))
        infeasibilities.append(v_k_new)
        solutions.append(xi_k.copy())
        rhos.append(rho)
        relative_diff_values.append(abs(objective_function(xi_k) - objective_function(p_k)) / max(1, abs(objective_function(p_k))))
        box_project_norms.append(np.linalg.norm(project_to_box(xi_k - approx_fprime(xi_k, lambda x: augmented_lagrangian(x, y, rho, obj_func), np.sqrt(np.finfo(float).eps))) - xi_k))

        if (relative_diff_values[-1] <= epsilon_s and box_project_norms[-1] > epsilon_a):
            print(f"Restarting at outer iteration {k+1}")
            H = np.diag(np.diag(H))
            continue

        if objective_function(xi_k) > objective_function(p_k) and v_k_new > v_k:
            print(f"Restarting due to increase in objective and infeasibility at outer iteration {k+1}")
            y = np.zeros(0)
            H = np.diag(np.diag(H))
            continue

        print(f"Outer iteration {k+1}, optimal solution: {xi_k}")

        if v_k_new < tol and relative_diff_values[-1] < epsilon_s:
            print(f"Outer iteration {k+1} converged.")
            break

        p_k = xi_k.copy()

    return p_k, objective_values, infeasibilities, rhos, relative_diff_values, box_project_norms

p0 = np.full(n, 1.2)

solution, objective_values, infeasibilities, rhos, relative_diff_values, box_project_norms = solnp_solver(objective_function)
print("Final optimal solution:", solution)
print(objective_values)
print(infeasibilities)
print(rhos)
print(relative_diff_values)
print(box_project_norms)

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
filename = f'rosen{n}_nf_nr.png'
plt.savefig(filename)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(relative_diff_values)), relative_diff_values, label='Relative Difference')
plt.xlabel('Outer Iteration')
plt.ylabel('Relative Difference')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(len(box_project_norms)), box_project_norms, label='Box Projection Norm')
plt.xlabel('Outer Iteration')
plt.ylabel('Box Projection Norm')
plt.legend()

plt.tight_layout()
filename = f'rosen{n}_relative_box_norm.png'
plt.savefig(filename)
