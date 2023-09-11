import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
import numpy as np
import os
import yaml
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.logger import logging

sns.set_style("darkgrid")


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

    save_dict = {}
    if out_dict.get("name") in ["asfw", "bpfw"]:
        save_dict = {
            "name": out_dict.get("name"),
            "center": out_dict.get("center").tolist(),
            "radius": out_dict.get("radius").tolist(),
            "Number of iterations": out_dict.get("number_iterations"),
            "active_set_size": int(out_dict.get("active_set_size_list")[-1]),
            "dual_function": float(out_dict.get("dual_list")[-1]),
            "dual_gap": float(out_dict.get("dual_gap_list")[-1]),
            "CPU_time": out_dict.get("CPU_time_list")[-1],
        }

    elif out_dict.get("name") == "appfw":
        save_dict = {
            "name": out_dict.get("name"),
            "center": out_dict.get("center").tolist(),
            "radius": out_dict.get("radius").tolist(),
            "Number of iterations": out_dict.get("number_iterations"),
            "active_set_size": int(out_dict.get("active_set_size_list")[-1]),
            "dual_function": float(out_dict.get("dual_list")[-1]),
            "delta": float(out_dict.get("delta_list")[-1]),
            "CPU_time": out_dict.get("CPU_time_list")[-1],
        }

    return save_dict


def print_on_console(out_dict):
    # out_dict : the output dictionary returned after algorithm training
    center = out_dict.get("center")
    radius = out_dict.get("radius")
    num_iterations = out_dict.get("number_iterations")
    active_set_size = out_dict.get("active_set_size_list")[-1]
    dual = out_dict.get("dual_list")[-1]
    CPU_time = out_dict.get("CPU_time_list")[-1]

    print(f"dual function = {dual:.3e}")

    if out_dict.get("name") in ["asfw", "bpfw"]:
        dual_gap = out_dict.get("dual_gap_list")[-1]
        print(f"dual gap = {dual_gap:.3e}")
    elif out_dict.get("name") == "appfw":
        delta = out_dict.get("delta_list")[-1]
        print(f"delta = {delta:.3e}")

    print(f"Number of non-zero components of x = {active_set_size}")
    print(f"Number of iterations = {num_iterations}")
    print(f"Total CPU time: {CPU_time}")
    print(f"center: {center} and radius: {radius} ")


def load_config(config_name, config_path):
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)
    return config


# Plotting

def plot_points_circle(A, r, c, title, path, show=True):
    # Separate x and y coordinates from A
    x_coords, y_coords = A[0], A[1]

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
    circle = Circle(c, radius=r, color='black', fill=False)
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


def plot_cpu_time_vs_dual_gap(cpu_time, dual_gap_values, algorithm_name, path, show=True):
    # plt.plot(cpu_time, dual_gap_values, marker='o', label=algorithm_name)
    sns.lineplot(x=cpu_time, y=dual_gap_values, marker='o', label=algorithm_name, errorbar=None)
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


def plot_active_set_size_vs_dual_gap(active_set_sizes, dual_gap_values, algorithm_name, path, show=True):
    # plt.plot(active_set_sizes, dual_gap_values, marker='o', label=algorithm_name)
    sns.lineplot(x=active_set_sizes, y=dual_gap_values, marker='o', label=algorithm_name, errorbar=None)
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


def plot_cpu_time_vs_objective_function(cpu_time, objective_function_values, algorithm_name, path, show=True):
    # plt.plot(cpu_time, objective_function_values, marker='o', label=algorithm_name)
    sns.lineplot(x=cpu_time, y=objective_function_values, marker='o', label=algorithm_name, errorbar=None)
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


def plot_iterations_vs_objective_function(iterations, objective_function_values, algorithm_name, path, show=True):
    # plt.plot(iterations, objective_function_values, marker='o', label=algorithm_name)
    sns.lineplot(x=iterations, y=objective_function_values, marker='o', label=algorithm_name, errorbar=None)
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


def plot_dual_gap_vs_iterations(iterations, dual_gap_values, algorithm_name, path, show=True):
    # plt.plot(iterations, dual_gap_values, marker='o', label=algorithm_name)
    sns.lineplot(x=iterations, y=dual_gap_values, marker='o', label=algorithm_name, errorbar=None)
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
    # plt.plot(cpu_time, delta_list, marker='o', label=algorithm_name)
    sns.lineplot(x=cpu_time, y=delta_list, marker='o', label=algorithm_name, errorbar=None)
    plt.xlabel('CPU Time')
    plt.ylabel('Delta')
    plt.title('CPU Time vs Delta')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(path, "cpu_time_vs_delta.png"))
    if show:
        plt.show()
    else:
        plt.close()


def plot_active_set_size_vs_delta(active_set_sizes, delta_list, algorithm_name, path, show=True):
    # plt.plot(active_set_sizes, delta_list, marker='o', label=algorithm_name)
    sns.lineplot(x=active_set_sizes, y=delta_list, marker='o', label=algorithm_name, errorbar=None)
    plt.xlabel('Size of Active Set')
    plt.ylabel('Delta')
    plt.title('Size of Active Set vs Delta')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(path, "active_set_size_vs_delta.png"))
    if show:
        plt.show()
    else:
        plt.close()


def plot_delta_vs_iterations(iterations, delta_list, algorithm_name, path, show=True):
    # plt.plot(iterations, delta_list, marker='o', label=algorithm_name)
    sns.lineplot(x=iterations, y=delta_list, marker='o', label=algorithm_name, errorbar=None)
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
        plot_dual_gap_vs_iterations(iterations_list, dual_gap_list, title, graph_path, show_graphs)
    elif out_dict.get("name") == "appfw":
        delta_list = out_dict.get("delta_list")
        plot_cpu_time_vs_delta(CPU_time_list, delta_list, title, graph_path, show_graphs)
        plot_active_set_size_vs_delta(active_set_size_list, delta_list, title, graph_path, show_graphs)
        plot_delta_vs_iterations(iterations_list, delta_list, title, graph_path, show_graphs)


# Data Generation
def create_data(data_config):
    data_creation_method = data_config.get('method')
    m = eval(data_config.get('number_of_samples'))
    n = eval(data_config.get('number_of_variables'))
    test_split = eval(data_config.get('test_split'))
    train_split = 1 - test_split

    if data_creation_method == "random_standard":

        train = generate_random_matrix_normal(0, 0.6, n * train_split, m * train_split)
        T_0 = generate_random_matrix_normal(0, 0.6, n * (test_split / 2), m * (test_split / 2))
        T_1 = generate_random_matrix_normal(0.6, 1, n * (test_split / 2), m * (test_split / 2))
        test_X = np.hstack((T_0, T_1))
        test_Y = [0] * len(T_0) + [1] * len(T_1)

    # elif data_creation_method == "fermat":
    #     # TODO: for Dejan -- what should be the logic to create fermat test and train
    #     train = generate_fermat_spiral(m).T
    # elif data_creation_method == "random_uniform":
    #     train = generate_random_matrix_uniform(0, 0.6, n*train_split, m*train_split)

    elif data_creation_method == "daphnet_freezing_data":
        train, test_X, test_Y = daphnet_freezing_data(test_split)

    elif data_creation_method == "metro_train_data":
        train, test_X, test_Y = metro_train_data(test_split)
    return train, (test_X, test_Y)


# daphnet_freezing_data
def daphnet_freezing_data(test_split):
    df = pd.read_csv('datasets/daphnet_freezing.arff', header=None)
    df = df.rename(columns={14: 'y'})

    # Create training and testing sets
    train, test = train_test_split(df, test_size=test_split)

    X = train[train.y == 0]

    # Remove generated row names and class column (y)
    X = X.drop(columns=['y'])

    # Normalize data
    scalar = MinMaxScaler()
    train_data = scalar.fit_transform(X).T

    len_good = (test["y"] == 0).sum()
    len_anomaly = (test["y"] == 1).sum()

    # Remove generated row names and class column (y)
    test = test.drop(columns=['y'])

    # normalize data
    test_X = scalar.fit_transform(test).T

    # create a list of zeros and ones for good and anomaly points
    test_Y = [0] * len_good + [1] * len_anomaly

    return train_data, test_X, test_Y


# metro_train_data
def metro_train_data(test_split):
    df = pd.read_csv('datasets/metro_train.csv')
    columns = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current', 'COMP']
    df = df[columns]
    df = df.rename(columns={"COMP": 'y'})

    # make anomaly as 1 and good as 0 (to make it suitable for algortihms)
    df['y'] = df['y'].replace({0: 1, 1: 0})

    # Create training and testing sets
    train, test = train_test_split(df, test_size=test_split)

    # Normalize data
    scalar = MinMaxScaler()

    X = train[train.y == 0]

    # Remove generated row names and class column (y)
    X = X.drop(columns=['y'])

    train_data = scalar.fit_transform(X).T

    len_good = (test["y"] == 0).sum()
    len_anomaly = (test["y"] == 0).sum()

    # Remove generated row names and class column (y)
    test = test.drop(columns=['y'])

    # normalize data
    test_X = scalar.fit_transform(test)

    # normalize data
    test_X = scalar.fit_transform(test).T

    # create a list of zeros and ones for good and anomaly points
    test_Y = [0] * len_good + [1] * len_anomaly

    return train_data, test_X, test_Y


def generate_fermat_spiral(dot):
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


def generate_random_matrix_normal(mu, sigma, m, n):
    return np.random.normal(loc=mu, scale=sigma, size=(m, n))


def generate_random_matrix_uniform(low, high, m, n):
    return np.random.uniform(low, high, size=(m, n))


# Algorithm Testing
def test_algorithm(test_data, center, radius):
    # test_data = (test_X, test_Y)
    # 0: good, 1: anomaly
    logging.info("\n----Testing Started------")

    true_negative = 0
    true_positive = 0
    false_negative = 0
    false_positive = 0
    test_X, test_Y = test_data
    number_of_test_points = test_X.shape[1]

    for point_idx in range(number_of_test_points):

        point = test_X[:, point_idx]
        y = test_Y[point_idx]
        assert y in [0, 1], "Variable must be either 0 or 1"
        dist = calculate_euc_distance(point, center)

        # logging.info(f"y:{y},dist>radius: {dist > radius},  distance: {dist}, radius {radius}")
        if y == 1:  # Then point is anomaly
            if dist > radius:  # then point is out of circle and it is anomaly
                true_positive += 1
                # logging.info(f"true positive ++, {true_positive}")
            else:  # Then point is inside circle but it is anomaly
                false_negative += 1
                # logging.info(f"false negative ++, {false_negative}")
        else:  # Then point is good
            if dist > radius:  # Then point is out of circle but it is good
                false_positive += 1
                # logging.info(f"false_positive ++, {false_positive}")
            else:  # Then point is in circle and it is good
                true_negative += 1
                # logging.info(f"true_negative ++, {true_negative}")

    logging.info(f"total points: {test_X.shape[1]}")
    logging.info(f"true positive: {true_positive}")
    logging.info(f"false negative: {false_negative}")
    logging.info(f"false positive: {false_positive}")
    logging.info(f"true_negative: {true_negative}")

    out_dict = {
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative
    }

    return create_test_save_dict(out_dict)


def create_test_save_dict(out_dict):
    TP = out_dict.get("tp")
    FP = out_dict.get("fp")
    TN = out_dict.get("tn")
    FN = out_dict.get("fn")

    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    save_dict = {
        "true_positive": TP,
        "true_negative": TN,
        "false_positive": FP,
        "false_negative": FN,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return save_dict


def calculate_euc_distance(point1, point2):
    return np.linalg.norm(point1 - point2)
