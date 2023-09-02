import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import os
import yaml
from pathlib import Path




# Create result save directory
# Create run folders for experiment result saving
def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

def create_save_dict(out_dict):
    # out_dict : the output dictionary returned after algorithm training
    save_dict = {
        "center": out_dict.get("center").tolist(),
        "radius": out_dict.get("radius").tolist(),
        "Number of iterations": out_dict.get("number_iterations"),
        "Last_active_set_size": int(out_dict.get("active_set_size_list")[-1]),
        "dual_function_list": [float(value) for value in out_dict.get("dual_list")],
        "dual_gap_list": [float(value) for value in out_dict.get("dual_gap_list")],
        "CPU_time": out_dict.get("CPU_time_list"),
    }
    return save_dict

def print_on_console(out_dict):
    # out_dict : the output dictionary returned after algorithm training
    center = out_dict.get("center")
    radius = out_dict.get("radius")
    num_iterations = out_dict.get("number_iterations")
    active_set_size_list = out_dict.get("active_set_size_list")
    dual_list = out_dict.get("dual_list")
    dual_gap_list = out_dict.get("dual_gap_list")
    CPU_time_list = out_dict.get("CPU_time_list")

    print(f"dual function = {dual_list[-1]:.3e}")
    print(f"dual gap = {dual_gap_list[-1]:.3e}")
    print(f"Number of non-zero components of x = {active_set_size_list[-1]}")
    print(f"Number of iterations = {num_iterations}")
    print(f"Total CPU time: {CPU_time_list[-1]}")
    print(f"center: {center} and radius: {radius} ")

def load_config(config_name,config_path):
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)

    return config


# Plotting
def plot_points_circle(A, r, c, title, path, show = True):
    # Separate x and y coordinates from A
    x_coords = A[0]
    y_coords = A[1]

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Plot the points as blue "+"
    ax.plot(x_coords, y_coords, 'b+', label='Inside points')

    # Plot the center as a blue thick dot
    ax.plot(c[0], c[1], 'bo', markersize=10, label='Center')

    # Plot the circle with black color
    circle = Circle(c, r, color='black', fill=False)
    ax.add_patch(circle)

    # Calculate distances from the center to each point
    distances = np.linalg.norm(A - c[:, np.newaxis], axis=0)
    # Find the indices of points that touch the boundary of the circle
    touching_indices = np.where(np.abs(distances - r) < 1e-6)[0]
    # Plot the points that touch the circle boundary as green "x"
    ax.plot(x_coords[touching_indices], y_coords[touching_indices], 'gx', label='Support vectors')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.set_title(title)
    # Set aspect ratio to be equal, so the circle isn't distorted
    ax.set_aspect('equal', adjustable='datalim')
    plt.savefig(os.path.join(path, "plot_points_circle.png"))
    if show:
        plt.show()
    else:
        plt.close()

def plot_test_data_and_circle(T,A, r, c, title, path, show = True):
    # Separate x and y coordinates from A
    x_coords = A[0]
    y_coords = A[1]

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Plot the points as blue "+"
    ax.plot(x_coords, y_coords, 'b+', label='Inside points')

    # Plot the center as a blue thick dot
    ax.plot(c[0], c[1], 'bo', markersize=10, label='Center')

    # Plot test points
    ax.plot(T[0], T[1], 'r*',label='test points')

    # Plot the circle with black color
    circle = Circle(c, radius = r, color='black', fill=False)
    ax.add_patch(circle)

    # Calculate distances from the center to each point
    distances = np.linalg.norm(A - c[:, np.newaxis], axis=0)
    # Find the indices of points that touch the boundary of the circle
    touching_indices = np.where(np.abs(distances - r) < 1e-6)[0]
    # Plot the points that touch the circle boundary as green "x"
    ax.plot(x_coords[touching_indices], y_coords[touching_indices], 'gx', label='Support vectors')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.set_title(title)
    # Set aspect ratio to be equal, so the circle isn't distorted
    ax.set_aspect('equal', adjustable='datalim')
    plt.savefig(os.path.join(path, "plot_points_circle.png"))
    if show:
        plt.show()
    else:
        plt.close()


def plot_cpu_time_vs_dual_gap(cpu_time, dual_gap_values, algorithm_name, path, show = True):
    plt.plot(cpu_time, dual_gap_values, marker='o', label=algorithm_name)
    plt.xlabel('CPU Time')
    plt.ylabel('Dual Gap')
    plt.title('CPU Time vs Dual Gap')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(path, "cpu_time_vs_dual_gap.png"))
    if show:
        plt.show()
    else:
        plt.close()

def plot_active_set_size_vs_dual_gap(active_set_sizes, dual_gap_values, algorithm_name, path,show = True):
    plt.plot(active_set_sizes, dual_gap_values, marker='o', label=algorithm_name)
    plt.xlabel('Size of Active Set')
    plt.ylabel('Dual Gap')
    plt.title('Size of Active Set vs Dual Gap')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(path, "active_set_size_vs_dual_gap.png"))
    if show:
        plt.show()
    else:
        plt.close()

def plot_cpu_time_vs_objective_function(cpu_time, objective_function_values, algorithm_name, path, show = True):
    plt.plot(cpu_time, objective_function_values, marker='o', label=algorithm_name)
    plt.xlabel('CPU Time')
    plt.ylabel('Objective Function Value')
    plt.title('CPU Time vs Objective Function')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(path, "cpu_time_vs_objective_function.png"))
    if show:
        plt.show()
    else:
        plt.close()

def plot_iterations_vs_objective_function(iterations, objective_function_values, algorithm_name, path, show = True):
    plt.plot(iterations, objective_function_values, marker='o', label=algorithm_name)
    plt.xlabel('Iterations')
    plt.ylabel('Objective Function Value')
    plt.title('Iterations vs Objective Function')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(path, "iterations_vs_objective_function.png"))
    if show:
        plt.show()
    else:
        plt.close()

def plot_dual_gap_vs_iterations(iterations, dual_gap_values, algorithm_name, path, show = True):
    plt.plot(iterations, dual_gap_values, marker='o', label=algorithm_name)
    plt.xlabel('Iterations')
    plt.ylabel('Dual Gap')
    plt.title('Dual Gap vs Iterations')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(path, "dual_gap_vs_iterations.png"))
    if show:
        plt.show()
    else:
        plt.close()


def plot_graphs(title, show_graphs, graph_path,out_dict):

    # out_dict : the output dictionary returned after algorithm training
    num_iterations = out_dict.get("number_iterations")
    active_set_size_list =out_dict.get("active_set_size_list")
    dual_list=out_dict.get("dual_list")
    dual_gap_list=out_dict.get("dual_gap_list")
    CPU_time_list=out_dict.get("CPU_time_list")

    os.mkdir(graph_path)
    iterations_list = list(range(num_iterations))
    # Plots to be showed/saved
    plot_cpu_time_vs_dual_gap(CPU_time_list, dual_gap_list, title, graph_path, show_graphs)
    plot_active_set_size_vs_dual_gap(active_set_size_list, dual_gap_list, title, graph_path, show_graphs)
    plot_cpu_time_vs_objective_function(CPU_time_list, dual_list, title, graph_path, show_graphs)
    plot_iterations_vs_objective_function(iterations_list, dual_list, title, graph_path, show_graphs)
    plot_dual_gap_vs_iterations(iterations_list, dual_gap_list, title, graph_path, show_graphs)



# Data Creation
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

def generateRandomMatrix(mu,sigma,n,m):
    return np.random.normal(loc=mu,scale=sigma, size= (n, m))


if __name__ == '__main__':

    maxiter = 1000
    epsilon = 1e-6

    # A_matrix = generateRandomMatrix(2 ** 3, 2 ** 1)
    # center, Xk_active_set, u_dual_sol, radius, total_time = one_plus_eps_MEB_approximation(A_matrix, epsilon, maxiter)

    f_spiral = fermat_spiral(8)
    # plt.scatter(f_spiral[len(f_spiral) // 2:, 0], f_spiral[len(f_spiral) // 2:, 1])
    # plt.show()
