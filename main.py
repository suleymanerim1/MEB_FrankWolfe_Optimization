import os
from src.logger import my_logger
from src.utils import increment_path, load_config, create_data, execute_algorithm, create_yaml, plot_comparison_graphs


# TODO (Marija): For every plot except the CPU time: x-axis should be a list of integers, however it shows floats.
# Problematic graphs: Size of Active Set vs Delta, Iterations vs objective Function, Delta vs Iterations
# This problem occurs when the number of iterations/size of active set is small (< 10).
# TODO (Marija): Check if a graph needs scaling and do when necessary
# TODO (Marija): comparison graphs, check and write (with graph show, save)
# in utils. I collected all graphs functions inside that.

# Change only this
yaml_name = "exp8_Thyroid.yaml"

config_path = "configs/"  # Folder to load config file
# Configuration
config = load_config(yaml_name, config_path)
show_graphs = config.get('show_graphs')
maxIter = eval(config.get('maxiter'))
epsilon = eval(config.get('epsilon'))
perform_test = config.get('perform_test')

# Save path
base_path = 'runs/'
experiment_path = os.path.join(base_path, os.path.splitext(yaml_name)[0])
# if there is an experiment with same experiment.yaml, increment_path_number exp1, exp2....
incremented_path = increment_path(experiment_path, exist_ok=False, sep='_', mkdir=True)
print(f"Results will be saved to: {incremented_path}")

# Create logger
logging = my_logger(incremented_path)
logging.info("Creating data points")

if __name__ == '__main__':

    # Initialize YAML dicts
    asfw_train, bpfw_train, appfw_train = {}, {}, {}
    asfw_test, bpfw_test, appfw_test = {}, {}, {}
    results = {}

    # Create Data
    A, test_data = create_data(config.get('data'))
    n, m = A.shape
    mc ={}
    # Start Algorithms
    solver_methods = config.get('solver_methods')
    for method in solver_methods:
        train, test = execute_algorithm(method, A, config, incremented_path, test_data)
        results[method] = (train, test)
        mc[method] = train

    plot_comparison_graphs(results)

    data_method = config.get("data").get("method")
    if data_method not in ["random_standard"]:
        del config["data"]["number_of_samples"]
        del config["data"]["number_of_variables"]

    if perform_test:
        test_size = len(test_data[1])
    else:
        test_size = 0

    # Create yaml content
    create_yaml(m, test_size, config, incremented_path,
                results.get('asfw', (None, None)),
                results.get('bpfw', (None, None)),
                results.get('appfw', (None, None)))
