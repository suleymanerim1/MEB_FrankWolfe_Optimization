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
    plt.plot(active_set_sizes, dual_gap_values, marker='o',label=algorithm_name)
    plt.xlabel('Size of Active Set')
    plt.ylabel('Dual Gap')
    plt.title('Size of Active Set vs Dual Gap')
    plt.grid(True)
    plt.show()


def plot_cpu_time_vs_objective_function(cpu_time, objective_function_values,algorithm_name):
    plt.plot(cpu_time, objective_function_values, marker='o',label = algorithm_name)
    plt.xlabel('CPU Time')
    plt.ylabel('Objective Function Value')
    plt.title('CPU Time vs Objective Function')
    plt.grid(True)
    plt.show()


def plot_iterations_vs_objective_function(iterations, objective_function_values, algorithm_name):
    plt.plot(iterations, objective_function_values, marker='o', label = algorithm_name)
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


dual_gap_values_away = [...]  # List of dual gap values for awayStep_FW
cpu_time_away = [...]  # Value of CPU time for awayStep_FW
active_set_sizes_away = [...]  # List of active set sizes for awayStep_FW
objective_function_values_away = [...]  # List of objective function values for awayStep_FW
iterations_away = list(range(1, len(objective_function_values_away) + 1)) # ??

dual_gap_values_blended = [...]  # List of dual gap values for blendedPairwise_FW
cpu_time_blended = [...]  # Value of CPU time for blendedPairwise_FW
active_set_sizes_blended = [...]  # List of active set sizes for blendedPairwise_FW
objective_function_values_blended = [...]  # List of objective function values for blendedPairwise_FW
iterations_blended = list(range(1, len(objective_function_values_blended) + 1)) # ??

# Plot graphs for awayStep_FW
plot_cpu_time_vs_dual_gap(cpu_time_away, dual_gap_values_away, 'awayStep_FW')
plot_active_set_size_vs_dual_gap(active_set_sizes_away, dual_gap_values_away, 'awayStep_FW')
plot_cpu_time_vs_objective_function(cpu_time_away, objective_function_values_away,'awayStep_FW')
plot_iterations_vs_objective_function(iterations_away, objective_function_values_away,'awayStep_FW')
plot_dual_gap_vs_iterations(iterations_away, dual_gap_values_away, 'awayStep_FW')

# Plot graphs for blendedPairwise_FW
plot_cpu_time_vs_dual_gap(cpu_time_blended, dual_gap_values_blended, 'blendedPairwise_FW')
plot_active_set_size_vs_dual_gap(active_set_sizes_blended, dual_gap_values_blended, 'blendedPairwise_FW')
plot_cpu_time_vs_objective_function(cpu_time_blended, objective_function_values_blended,'blendedPairwise_FW')
plot_iterations_vs_objective_function(iterations_blended, objective_function_values_blended,'blendedPairwise_FW')
plot_dual_gap_vs_iterations(iterations_blended, dual_gap_values_blended, 'blendedPairwise_FW')