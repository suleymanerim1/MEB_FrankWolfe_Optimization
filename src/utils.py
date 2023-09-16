import os
from pathlib import Path
import yaml

from src.logger import logging


def load_config(config_name, config_path):
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)
    return config


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


def create_yaml(train_size, test_size, config, save_path, results_dict):
    asfw_train, bpfw_train, appfw_train = {}, {}, {}
    asfw_test, bpfw_test, appfw_test = {}, {}, {}

    solver_methods = config.get("solver_methods")
    if "asfw" in solver_methods:
        asfw_train = results_dict.get("asfw")[0]
        asfw_train = __return_yaml_train_format(asfw_train)
        asfw_test = results_dict.get("asfw")[1]

    if "bpfw" in solver_methods:
        bpfw_train = results_dict.get("bpfw")[0]
        bpfw_train = __return_yaml_train_format(bpfw_train)
        bpfw_test = results_dict.get("bpfw")[1]

    if "appfw" in solver_methods:
        appfw_train = results_dict.get("appfw")[0]
        appfw_train = __return_yaml_train_format(appfw_train)
        appfw_test = results_dict.get("appfw")[1]

    data_method = config.get("data").get("method")
    if data_method not in ["random_standard", "random_uniform"]:
        del config["data"]["number_of_samples"]
        del config["data"]["number_of_variables"]

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
    with open(os.path.join(save_path, 'output.yaml'), 'w') as file:
        yaml.dump(output, file, sort_keys=False)
        logging.info(f"Output.yaml created")


def __return_yaml_train_format(out_dict):
    # out_dict : the output dictionary returned after algorithm training

    yaml_dict = {}
    if out_dict.get("name") in ["asfw", "bpfw"]:
        yaml_dict = {
            "name": out_dict.get("name"),
            "center": out_dict.get("center"),
            "radius": round(out_dict.get("radius"), 6),
            "Number of iterations": out_dict.get("Number of iterations"),
            "active_set_size": int(out_dict.get("active_set_size")[-1]),
            "dual_function": float(out_dict.get("dual_function")[-1]),
            "dual_gap": float(out_dict.get("dual_gap")[-1]),
            "CPU_time": round(out_dict.get("CPU_time")[-1]*1000, 3),
        }

    elif out_dict.get("name") == "appfw":
        yaml_dict = {
            "name": out_dict.get("name"),
            "center": out_dict.get("center"),
            "radius": round(out_dict.get("radius"), 6),
            "Number of iterations": out_dict.get("Number of iterations"),
            "active_set_size": int(out_dict.get("active_set_size")[-1]),
            "dual_function": float(out_dict.get("dual_function")[-1]),
            "delta": float(out_dict.get("delta")[-1]),
            "CPU_time": round(out_dict.get("CPU_time")[-1]*1000, 3),
        }

    return yaml_dict
