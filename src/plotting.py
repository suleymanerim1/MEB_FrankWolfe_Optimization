import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from deprecated import deprecated


sns.set_style("darkgrid")


# Plotting
def plot_points_circle(A, r, c, title, path, test_data=None, show=True):
    # Separate x and y coordinates from A
    x_coords, y_coords = A[0], A[1]

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Plot the points as blue "+"
    ax.plot(x_coords, y_coords, 'b+', label='Inside points')

    if test_data[0] is not None:
        test_X = test_data[0]
        x_coords_test, y_coords_test = test_X[0], test_X[1]
        # Plot the anomalies as black "+"
        ax.plot(x_coords_test, y_coords_test, 'r+', label='Test points')

    # Plot the center as a cyan thick dot
    ax.plot(c[0], c[1], 'co', markersize=10, label='Center')

    # Plot the circle with black color
    circle = Circle(c, r, color='black', linewidth=2, fill=False)
    ax.add_patch(circle)

    # Calculate distances from the center to each point
    distances = np.linalg.norm(A - c[:, np.newaxis], axis=0)
    # Find the indices of points that touch the boundary of the circle
    touching_indices = np.where(np.abs(distances - r) < 1e-6)[0]
    # Plot the points that touch the circle boundary as green "x"
    ax.plot(x_coords[touching_indices], y_coords[touching_indices], 'gX', markersize=10, label='Support vectors')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.set_title(title)
    # Set aspect ratio to be equal, so the circle isn't distorted
    ax.set_aspect('equal', adjustable='datalim')
    plt.savefig(os.path.join(path, "points_circle.png"))
    if show:
        plt.show()
    else:
        plt.close()

def plot_comparison_graphs(out_dict, show_graphs, graph_path):
    train_dict = {key: value[0] for key, value in out_dict.items()}
    plot_single_comparison_graph(train_dict, 'CPU_time', 'dual_function',
                                 'CPU time', 'Objective function', graph_path, show_graphs)
    plot_single_comparison_graph(train_dict, 'active_set_size', 'dual_function',
                                 'Active Set Size', 'Objective function', graph_path, show_graphs)
    plot_single_comparison_graph(train_dict, "Number of iterations", 'dual_function',
                                 'Number of iterations', 'Objective function', graph_path, show_graphs)
    plot_single_comparison_graph(train_dict, "Number of iterations", 'dual_gap',
                                 'Number of iterations', 'Dual Gap', graph_path, show_graphs)
    plot_single_comparison_graph(train_dict, "CPU_time", 'dual_gap',
                                 'CPU time', 'Dual Gap', graph_path, show_graphs)
    plot_single_comparison_graph(train_dict, 'active_set_size', 'dual_gap',
                                 'Active Set Size', 'Dual gap', graph_path, show_graphs)
    plot_single_comparison_graph(train_dict, "CPU_time", 'active_set_size',
                                 'CPU time', 'Active set size', graph_path, show_graphs)
    plot_single_comparison_graph(train_dict, 'Number of iterations', 'active_set_size',
                                 'Number of iterations', 'Active set size', graph_path, show_graphs)

def plot_graphs(title, show_graphs, graph_path, out_dict):
    # out_dict : the output dictionary returned after algorithm training
    num_iterations = out_dict.get("number_iterations")
    active_set_size_list = out_dict.get("active_set_size_list")
    dual_list = out_dict.get("dual_list")
    CPU_time_list = out_dict.get("CPU_time_list")

    os.mkdir(graph_path)
    iterations_list = list(range(num_iterations))

    plot_cpu_time_vs_objective_function(CPU_time_list, dual_list, title, graph_path, show_graphs)
    plot_iterations_vs_objective_function(iterations_list, dual_list, title, graph_path, show_graphs)

    # Plots to be showed/saved
    if out_dict.get("name") in ["asfw", "bpfw"]:
        dual_gap_list = out_dict.get("dual_gap_list")
        plot_cpu_time_vs_dual_gap(CPU_time_list, dual_gap_list, title, graph_path, show_graphs)
        plot_active_set_size_vs_dual_gap(active_set_size_list, dual_gap_list, title, graph_path, show_graphs)
        plot_iterations_vs_dual_gap(iterations_list, dual_gap_list, title, graph_path, show_graphs)
    elif out_dict.get("name") == "appfw":
        delta_list = out_dict.get("delta_list")
        plot_cpu_time_vs_delta(CPU_time_list, delta_list, title, graph_path, show_graphs)
        plot_active_set_size_vs_delta(active_set_size_list, delta_list, title, graph_path, show_graphs)
        plot_iterations_vs_delta(iterations_list, delta_list, title, graph_path, show_graphs)



def plot_cpu_time_vs_dual_gap(cpu_time, dual_gap_values, algorithm_name, path, show=True):
    plt.plot(cpu_time, dual_gap_values, label=algorithm_name)
    # sns.lineplot(x=cpu_time, y=dual_gap_values, label=algorithm_name, errorbar=None)
    plt.xlabel('CPU Time')
    plt.ylabel('Dual Gap')
    plt.title('Dual Gap vs CPU Time')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(path, "dual_gap_vs_cpu_time.png"))
    if show:
        plt.show()
    else:
        plt.close()


def plot_active_set_size_vs_dual_gap(active_set_sizes, dual_gap_values, algorithm_name, path, show=True):
    plt.plot(active_set_sizes, dual_gap_values, label=algorithm_name)
    # sns.lineplot(x=active_set_sizes, y=dual_gap_values, label=algorithm_name, errorbar=None)
    plt.xlabel('Size of Active Set')
    plt.ylabel('Dual Gap')
    plt.title('Dual Gap vs Size of Active Set')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(path, "dual_gap_vs_active_set_size.png"))
    if show:
        plt.show()
    else:
        plt.close()


def plot_cpu_time_vs_objective_function(cpu_time, objective_function_values, algorithm_name, path, show=True):
    plt.plot(cpu_time, objective_function_values, label=algorithm_name)
    # sns.lineplot(x=cpu_time, y=objective_function_values, label=algorithm_name, errorbar=None)
    plt.xlabel('CPU Time')
    plt.ylabel('Objective Function Value')
    plt.title('Objective Function vs CPU Time')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(path, "objective_function_vs_cpu_time.png"))
    if show:
        plt.show()
    else:
        plt.close()


def plot_iterations_vs_objective_function(iterations, objective_function_values, algorithm_name, path, show=True):
    plt.plot(iterations, objective_function_values, label=algorithm_name)
    # sns.lineplot(x=iterations, y=objective_function_values, label=algorithm_name, errorbar=None)
    plt.xlabel('Iterations')
    plt.ylabel('Objective Function Value')
    plt.title('Objective Function vs Iterations')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(path, "objective_function_vs_iterations.png"))
    if show:
        plt.show()
    else:
        plt.close()


def plot_iterations_vs_dual_gap(iterations, dual_gap_values, algorithm_name, path, show=True):
    plt.plot(iterations, dual_gap_values, label=algorithm_name)
    # sns.lineplot(x=iterations, y=dual_gap_values, label=algorithm_name, errorbar=None)
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


def plot_cpu_time_vs_delta(cpu_time, delta_list, algorithm_name, path, show=True):
    plt.plot(cpu_time, delta_list, label=algorithm_name)
    # sns.lineplot(x=cpu_time, y=delta_list, label=algorithm_name, errorbar=None)
    plt.xlabel('CPU Time')
    plt.ylabel('Delta')
    plt.title('Delta_vs_CPU Time')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(path, "delta_vs_cpu_time.png"))
    if show:
        plt.show()
    else:
        plt.close()


def plot_active_set_size_vs_delta(active_set_sizes, delta_list, algorithm_name, path, show=True):
    plt.plot(active_set_sizes, delta_list, label=algorithm_name)
    # sns.lineplot(x=active_set_sizes, y=delta_list, label=algorithm_name, errorbar=None)
    plt.xlabel('Size of Active Set')
    plt.ylabel('Delta')
    plt.title('Delta vs Size of Active Set')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(path, "delta_vs_active_set_size.png"))
    if show:
        plt.show()
    else:
        plt.close()


def plot_iterations_vs_delta(iterations, delta_list, algorithm_name, path, show=True):
    plt.plot(iterations, delta_list, label=algorithm_name)
    # sns.lineplot(x=iterations, y=delta_list, label=algorithm_name, errorbar=None)
    plt.xlabel('Iterations')
    plt.ylabel('Delta')
    plt.title('Delta vs Iterations')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(path, "delta_vs_iterations.png"))
    if show:
        plt.show()
    else:
        plt.close()


def plot_single_comparison_graph(train_dict, x_string, y_string, x_label, y_label, path, show=False, transform=None):
    plt.plot()
    for key, value in train_dict.items():
        if x_string == "Number of iterations":
            x_axis = list(range(value[x_string]))
        else:
            x_axis = value[x_string]

        if y_string == "dual_gap" and key == "appfw":
            y_string = "delta"

        line_style = '--' if key == 'bpfw' else '-'

        plt.plot(x_axis, value[y_string], linewidth=2,linestyle=line_style, label=key.upper(),marker="o",ms=7,markevery=[-1])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    title = y_label + " vs " + x_label
    plt.title(title)
    plt.legend()

    if transform == 'log':
        plt.yscale('log')  # Set y-axis to logarithmic scale
    elif transform == 'exp':
        plt.yscale('exp')

    plt.savefig(os.path.join(path, title + ".png"))
    if show:
        plt.show()
    else:
        plt.close()





@deprecated(reason="This method is deprecated. Use plot_points_circle() instead.")
def plot_test_data_and_circle(T, A, r, c, title, path, show=True):
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
    ax.plot(T[0], T[1], 'r*', label='test points')

    # Plot the circle with black color
    circle = Circle(c, radius=r, color='black', linewidth=5, fill=False)
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