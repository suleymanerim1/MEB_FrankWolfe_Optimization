import yaml
import os
from src.logger import my_logger
from src.FrankWolfeVariants import awayStep_FW, blendedPairwise_FW, one_plus_eps_MEB_approximation
from src.utils import increment_path, load_config, plot_points_circle
from src.utils import create_data, create_save_dict, print_on_console, plot_graphs, test_algorithm


# Change only this
yaml_name = "exp1.yaml"

# Folder to load config file
config_path = "configs/"
# Configuration
config = load_config(yaml_name, config_path)
show_graphs = config.get('show_graphs')
maxiter = eval(config.get('maxiter'))
epsilon = eval(config.get('epsilon'))
test = config.get('test')

# TODO: create functions to automatically arrange real data for train and testing

# TODO: find 2 datasets to check -- for Marija
# TODO: comparison graphs, check and write (with graph show, save) -- for Marija
# in utils. I collected all graphs functions inside that.


# Save path
base_path = 'runs/'
experiment_path = os.path.join(base_path, os.path.splitext(yaml_name)[0])
# if there is an experiment with same experiment.yaml, increment_path_number exp1, exp2....
incremented_path = increment_path(experiment_path, exist_ok=False, sep='_', mkdir=True)
print(f"Results will be saved: {incremented_path}")

# Create logger
logging = my_logger(incremented_path)
logging.info("Creating data points")

# Create Data
A,test_data = create_data(config.get('data'))
n, m = A.shape

if __name__ == '__main__':

    # Initialize YAML dicts
    asfw_train = {}
    bpfw_train = {}
    appfw_train = {}
    if test:
        asfw_test = {}
        bpfw_test = {}
        appfw_test = {}

    # Start Algorithms
    solver_methods = config.get('solver_methods')
    for method in solver_methods:

        if method == "asfw":
            print("\n*****************")
            title = "*  Away Step FW   *"
            print(title)
            print("*****************")

            logging.info("\nASFW algorithm started!")
            line_search_strategy = config.get('line_search_asfw')
            out_dict = awayStep_FW(A, epsilon, line_search_strategy, maxiter)

            # Print results:
            print_on_console(out_dict)

            # Create dict to save results on YAML
            asfw_train = create_save_dict(out_dict)
            # Plot graphs for awayStep_FW
            graph_path = os.path.join(incremented_path, "asfw_graphs")
            plot_graphs(title, show_graphs, graph_path, out_dict)
            if n == 2:
                plot_points_circle(A, out_dict.get("radius"), out_dict.get("center"), title, graph_path, show_graphs)

            # test_data,center, radius
            if test:
                logging.info("ASFW Test")
                asfw_test = test_algorithm(test_data,out_dict.get("center"), out_dict.get("radius"))

        if method == "bpfw":
            print("\n*****************")
            title = "*  Blended Pairwise FW   *"
            print(title)
            print("*****************")

            logging.info("\nBPFW algorithm started!")
            line_search_strategy = config.get('line_search_bpfw')
            out_dict = blendedPairwise_FW(A, epsilon, line_search_strategy, maxiter)

            # Print results:
            print_on_console(out_dict)

            # Create dict to save results on YAML
            bpfw_train = create_save_dict(out_dict)

            # Plot graphs for blendedPairwise_FW
            graph_path = os.path.join(incremented_path, "bpfw_graphs")
            plot_graphs(title, show_graphs, graph_path, out_dict)
            if n == 2:
                plot_points_circle(A, out_dict.get("radius"), out_dict.get("center"), title, graph_path, show_graphs)

            # test_data,center, radius
            if test:
                logging.info("BPFW Test")
                bpfw_test = test_algorithm(test_data,out_dict.get("center"), out_dict.get("radius"))

        if method == "appfw":
            print("\n*****************")
            title = "*  (1+epsilon)-approximation FW   *"
            print(title)
            print("*****************")
            logging.info("\n(1+epsilon)-approximation FW algorithm started!")

            out_dict = one_plus_eps_MEB_approximation(A, epsilon, maxiter)

            # Print results:
            print_on_console(out_dict)

            # create dict to save results on YAML
            appfw_train = create_save_dict(out_dict)

            # Plot graphs for one_plus_eps_MEB_approximation
            graph_path = os.path.join(incremented_path, "appfw_graphs")
            plot_graphs(title, show_graphs, graph_path, out_dict)
            if n == 2:
                plot_points_circle(A, out_dict.get("radius"), out_dict.get("center"), title, graph_path, show_graphs)

            # test_data,center, radius
            if test:
                logging.info("APPFW Test")
                appfw_test = test_algorithm(test_data,out_dict.get("center"), out_dict.get("radius"))





    data_method = config.get("data").get("method")
    if data_method not in ["random_standard"]:
        del config["data"]["number_of_samples"]
        del config["data"]["number_of_variables"]

    # Create yaml content
    output = {
        'train':
            {
                'train_points': m,
                'asfw': asfw_train,
                'bpfw': bpfw_train,
                'appfw': appfw_train,
            },
        'test':
            {
                'test_points': len(test_data[1]),
                'asfw': asfw_test,
                'bpfw': bpfw_test,
                'appfw': appfw_test,
            },
        'config': config
    }


    # Save output yaml file
    with open(os.path.join(incremented_path, 'output.yaml'), 'w') as file:
        yaml.dump(output, file, sort_keys = False)
        logging.info(f"Output.yaml created")
