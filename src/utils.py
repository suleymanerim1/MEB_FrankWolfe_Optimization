import math
import matplotlib.pyplot as plt
import numpy as np


def plot_cpu_time_vs_dual_gap(cpu_time, dual_gap_values, algorithm_name):
    plt.plot(cpu_time, dual_gap_values, marker='o', label=algorithm_name)
    plt.xlabel('CPU Time')
    plt.ylabel('Dual Gap')
    plt.title('CPU Time vs Dual Gap')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_active_set_size_vs_dual_gap(active_set_sizes, dual_gap_values, algorithm_name):
    plt.plot(active_set_sizes, dual_gap_values, marker='o', label=algorithm_name)
    plt.xlabel('Size of Active Set')
    plt.ylabel('Dual Gap')
    plt.title('Size of Active Set vs Dual Gap')
    plt.grid(True)
    plt.show()

def plot_cpu_time_vs_objective_function(cpu_time, objective_function_values, algorithm_name):
    plt.plot(cpu_time, objective_function_values, marker='o', label=algorithm_name)
    plt.xlabel('CPU Time')
    plt.ylabel('Objective Function Value')
    plt.title('CPU Time vs Objective Function')
    plt.grid(True)
    plt.show()

def plot_iterations_vs_objective_function(iterations, objective_function_values, algorithm_name):
    plt.plot(iterations, objective_function_values, marker='o', label=algorithm_name)
    plt.xlabel('Iterations')
    plt.ylabel('Objective Function Value')
    plt.title('Iterations vs Objective Function')
    plt.grid(True)
    plt.show()

def plot_dual_gap_vs_iterations(iterations, dual_gap_values, algorithm_name):
    plt.plot(iterations, dual_gap_values, marker='o', label=algorithm_name)
    plt.xlabel('Iterations')
    plt.ylabel('Dual Gap')
    plt.title('Dual Gap vs Iterations')
    plt.grid(True)
    plt.legend()
    plt.show()

def fermat_spiral(dot):
    data = []
    d = dot * 0.1
    for i in range(dot):
        t = i / d * np.pi
        x = (1 + t) * math.cos(t)
        y = (1 + t) * math.sin(t)
        data.append([x, y])
    narr = np.array(data)
    f_s = np.concatenate((narr, -narr))
    return f_s

def generateRandomMatrix(n, m):
    return np.random.randn(n, m)


if __name__ == '__main__':

    maxiter = 1000
    epsilon = 1e-6

    # A_matrix = generateRandomMatrix(2 ** 3, 2 ** 1)
    # center, Xk_active_set, u_dual_sol, radius, total_time = one_plus_eps_MEB_approximation(A_matrix, epsilon, maxiter)

    f_spiral = fermat_spiral(8)
    # plt.scatter(f_spiral[len(f_spiral) // 2:, 0], f_spiral[len(f_spiral) // 2:, 1])
    # plt.show()
