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
from src.FrankWolfeVariants import awayStep_FW, blendedPairwise_FW, one_plus_eps_MEB_approximation
from deprecated import deprecated
import scipy.io as io


sns.set_style("darkgrid")


# Create result save directory
# Create run folders for experiment result saving
def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    p = ""
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

@deprecated(reason="This method is deprecated. Use create_save_dict_full() instead.")
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


def create_save_dict_full(out_dict):
    # out_dict : the output dictionary returned after algorithm training

    save_dict = {}
    if out_dict.get("name") in ["asfw", "bpfw"]:
        save_dict = {
            "name": out_dict.get("name"),
            "center": out_dict.get("center").tolist(),
            "radius": out_dict.get("radius").tolist(),
            "Number of iterations": out_dict.get("number_iterations"),
            "active_set_size": (out_dict.get("active_set_size_list")),
            "dual_function": (out_dict.get("dual_list")),
            "dual_gap": (out_dict.get("dual_gap_list")),
            "CPU_time": out_dict.get("CPU_time_list"),
        }

    elif out_dict.get("name") == "appfw":
        save_dict = {
            "name": out_dict.get("name"),
            "center": out_dict.get("center").tolist(),
            "radius": out_dict.get("radius").tolist(),
            "Number of iterations": out_dict.get("number_iterations"),
            "active_set_size": (out_dict.get("active_set_size_list")),
            "dual_function": (out_dict.get("dual_list")),
            "delta": (out_dict.get("delta_list")),
            "CPU_time": out_dict.get("CPU_time_list"),
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
    plt.plot(iterations, np.log(dual_gap_values), label=algorithm_name)
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


def plot_single_comparison_graph(train_dict, x_string, y_string, x_label, y_label, path, show=False):
    plt.plot()
    for key, value in train_dict.items():
        if x_string == "Number of iterations":
            x_axis = list(range(value[x_string]))
        else:
            x_axis = value[x_string]

        if y_string == "dual_gap" and key == "appfw":
            y_string = "delta"

        plt.plot(x_axis, value[y_string], linewidth=2, label=key)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    title = y_label + " " + "x_label"
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(path, title + ".png"))
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


# Data Generation
def create_data(data_config):
    data_creation_method = data_config.get('method')
    m = eval(data_config.get('number_of_samples'))
    n = eval(data_config.get('number_of_variables'))
    test_split = eval(data_config.get('test_split'))
    train_split = 1 - test_split
    train, test_X, test_Y = None, None, None

    if data_creation_method == "random_normal":
        train = generate_random_matrix_normal(0, 1, n, int(m * train_split))
        T_0 = generate_random_matrix_normal(0, 1, n, int(m * test_split * 0.5))
        T_1 = generate_random_matrix_normal(8, 1, n, int(m * test_split * 0.5))
        test_X = np.hstack((T_0, T_1))
        test_Y = [0] * int(m * test_split * 0.5) + [1] * int(m * test_split * 0.5)

    elif data_creation_method == "fermat_spiral":
        train = generate_fermat_spiral(m // 2).T

    elif data_creation_method == "random_uniform":
        train = generate_random_matrix_uniform(0, 0.7, n, int(m * train_split))
        test_X = generate_random_matrix_uniform(0.7, 1, n, int(m * test_split))
        test_Y = [1] * int(m * test_split)

    elif data_creation_method == "daphnet_freezing_data":
        train, test_X, test_Y = daphnet_freezing_data(test_split)

    elif data_creation_method == "metro_train_data":
        train, test_X, test_Y = metro_train_data(test_split)

    elif data_creation_method == "thyroid_data":
        train, test_X, test_Y = thyroid_data(test_split)

    elif data_creation_method == 'breast_cancer_data':
        train, test_X, test_Y = breast_cancer_data(test_split)

    return train, (test_X, test_Y)


# daphnet_freezing_data
def daphnet_freezing_data(test_split, seed=123):
    np.random.seed(seed)

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


def breast_cancer_data(test_split, seed=123):
    np.random.seed(seed)

    medical_df = pd.read_csv('datasets/breast_cancer_wisconsin.csv')
    labels = medical_df['diagnosis']
    labels = labels.map({'B': 0, 'M': 1})
    features_to_drop = ['Unnamed: 32', 'id', 'diagnosis']
    features = medical_df.drop(features_to_drop, axis=1)

    nominal_data = features.loc[labels == 0, :]
    nominal_labels = labels[labels == 0]
    N_nominal = nominal_data.shape[0]

    anomaly_data = features.loc[labels == 1, :]
    anomaly_labels = labels[labels == 1]

    randIdx = np.arange(N_nominal)
    np.random.shuffle(randIdx)

    N_train = int(N_nominal * (1 - test_split))

    # (1 - test_split) nominal data as training set
    X_train = nominal_data.iloc[randIdx[:N_train]].values

    # test_split nominal data + all novel data as test set
    X_test = np.concatenate((nominal_data.iloc[randIdx[N_train:]], anomaly_data), axis=0)
    y_test = np.concatenate((nominal_labels.iloc[randIdx[N_train:]], anomaly_labels), axis=0)

    return X_train.T, X_test.T, y_test


def thyroid_data(test_split, seed=123):
    np.random.seed(seed)

    data = io.loadmat('datasets/thyroid.mat')

    features = data['X']
    labels = data['y'].squeeze()
    labels = (labels == 0)

    nominal_data = features[labels == 1, :]
    nominal_labels = labels[labels == 1]

    N_nominal = nominal_data.shape[0]

    anomaly_data = features[labels == 0, :]
    anomaly_labels = labels[labels == 0]

    randIdx = np.arange(N_nominal)
    np.random.shuffle(randIdx)

    N_train = int(N_nominal * (1 - test_split))

    # 0.5 nominal data as training set
    X_train = nominal_data[randIdx[:N_train]]

    # 0.5 nominal data + all novel data as test set
    X_test = nominal_data[randIdx[N_train:]]
    y_test = nominal_labels[randIdx[N_train:]]
    X_test = np.concatenate((X_test, anomaly_data), axis=0)
    y_test = np.concatenate((y_test, anomaly_labels), axis=0)

    return X_train.T, X_test.T, y_test


# metro_train_data
def metro_train_data(test_split):
    df = pd.read_csv('datasets/metro_train.csv')
    columns = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current', 'COMP']
    df = df[columns]
    df = df.rename(columns={"COMP": 'y'})

    # make anomaly as 1 and good as 0 (to make it suitable for algorithms)
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
    test_X = scalar.fit_transform(test).T

    # create a list of zeros and ones for good and anomaly points
    test_Y = [0] * len_good + [1] * len_anomaly

    return train_data, test_X, test_Y


def generate_fermat_spiral(dot, seed=123):
    np.random.seed(seed)
    data = []
    d = dot * 0.1
    for i in range(dot):
        t = i / d * np.pi
        x = (1 + t) * math.cos(t)
        y = (1 + t) * math.sin(t)
        data.append([x, y])
    narr = np.array(data)
    f_s = np.concatenate((narr, -narr))
    np.random.shuffle(f_s)
    return f_s


def generate_random_matrix_normal(mu, sigma, m, n, seed=123):
    np.random.seed(seed)
    return np.random.normal(loc=mu, scale=sigma, size=(m, n))


def generate_random_matrix_uniform(low, high, m, n, seed=123):
    np.random.seed(seed)
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


def execute_algorithm(method, A, config, incremented_path, test_data=None):

    show_graphs = config.get('show_graphs')
    maxIter = eval(config.get('maxiter'))
    epsilon = eval(config.get('epsilon'))
    perform_test = config.get('perform_test')

    print("\n*****************")
    title = method.upper()
    print(title)
    print("*****************")

    logging.info("\n"+method+" algorithm started!")
    out_dict = {}
    if method == "asfw":
        line_search_strategy = config.get('line_search_asfw')
        out_dict = awayStep_FW(A, epsilon, line_search_strategy, maxIter)
    elif method == "bpfw":
        line_search_strategy = config.get('line_search_bpfw')
        out_dict = blendedPairwise_FW(A, epsilon, line_search_strategy, maxIter)
    elif method == "appfw":
        out_dict = one_plus_eps_MEB_approximation(A, epsilon, maxIter)

    # Print results:
    print_on_console(out_dict)

    # Create dict to save results on YAML
    train_dict = create_save_dict_full(out_dict)
    # Plot graphs for awayStep_FW
    graph_path = os.path.join(incremented_path, method+"_graphs")
    plot_graphs(title, show_graphs, graph_path, out_dict)
    if A.shape[0] == 2:
        plot_points_circle(A, out_dict.get("radius"), out_dict.get("center"), title, graph_path, test_data, show_graphs)

    test_dict = None
    # test_data, center, radius
    if perform_test:
        logging.info(method+" Test")
        test_dict = test_algorithm(test_data, out_dict.get("center"), out_dict.get("radius"))

    return train_dict, test_dict


def create_yaml(train_size, test_size, config, incremented_path,
                asfw_train=None, asfw_test=None, bpfw_train=None, bpfw_test=None, appfw_train=None, appfw_test=None):
    output = {
        'train':
            {
                'train_points': train_size,
                'asfw': asfw_train,
                'bpfw': bpfw_train,
                'appfw': appfw_train,
            },
        'test':
            {
                'test_points': test_size,
                'asfw': asfw_test,
                'bpfw': bpfw_test,
                'appfw': appfw_test,
            },
        'config': config
    }

    # Save output yaml file
    with open(os.path.join(incremented_path, 'output.yaml'), 'w') as file:
        yaml.dump(output, file, sort_keys=False)
        logging.info(f"Output.yaml created")
