import numpy as np
import yaml
import os
from src.logger import my_logger
from src.FrankWolfeVariants import awayStep_FW, blendedPairwise_FW, one_plus_eps_MEB_approximation
from src.utils import increment_path, load_config, generateRandomMatrix, plot_points_circle, fermat_spiral
from src.utils import plot_cpu_time_vs_dual_gap, plot_active_set_size_vs_dual_gap, plot_cpu_time_vs_objective_function, \
    plot_iterations_vs_objective_function, plot_dual_gap_vs_iterations

# Change only this
yaml_name = "exp1.yaml"

# folder to load config file
config_path = "configs/"
# hyperparameters
config = load_config(yaml_name, config_path)
maxiter = eval(config.get('maxiter'))
epsilon = eval(config.get('epsilon'))
m = eval(config.get('number_of_samples'))
n = eval(config.get('number_of_variables'))
solver_methods = config.get('solver_methods')
data_creation_method = config.get('data_creation_method')
show_graphs = config.get('show_graphs')


# TODO: solve MEB problem and find anomaly --points for Suleyman
# TODO: keep track of anomaly points --for Suleyman
# TODO: important --- 3rd algorithm always stops at first iteration eps condition, check why?? -- for Suleyman
# TODO: 2nd algoritm return negative and positive, check that it creates error --for Suleyman
# TODO: Add other line_search_strategies --for Suleyman

# TODO: find 2 datasets to check -- for Marija
# TODO: change size of active set in graph from float to int -- for Marija
# TODO: comparison graphs, check and write (with graph show, save) -- for Marija


if __name__ == '__main__':

    # Save path
    base_path = 'runs/'
    experiment_path = os.path.join(base_path, os.path.splitext(yaml_name)[0])
    # if there is an experiment with same experiment.yaml, increment_path_number exp1, exp2....
    incremented_path = increment_path(experiment_path, exist_ok=False, sep='_', mkdir=True)
    print(f"Results will be saved: {incremented_path}")

    logging = my_logger(incremented_path)
    # maxiter = 1000
    # epsilon = 1e-6

    # m = 2 ** 10  # Number of samples
    # n = 2 ** 4  # Dimension of variables

    # methods = ["asfw", "bpfw", "appfw"]
    # methods = ["appfw"]

    logging.info("Creating data points")
    if data_creation_method == "random":
        A = generateRandomMatrix(n, m)
    elif data_creation_method == "fermat":
        A = fermat_spiral(m).T
    else:
        pass  # TODO: choose 2 datasets and add in this method

    asfw = {}
    bpfw = {}
    appfw = {}
    for method in solver_methods:
        # initialize output dictionaries

        if method == "asfw":
            print("\n*****************")
            title = "*  Away Step FW   *"
            print(title)
            print("*****************")

            logging.info("\nASFW algorithm started!")
            (center_awayStep_FW,
             radius_awayStep_FW,
             active_set_size_list_fw,
             dual_gap_list_fw,
             dual_list_fw,
             CPU_time_list_fw) = awayStep_FW(A, epsilon, maxiter)

            # Print results:
            num_iterations_fw = len(dual_list_fw)
            print(f"dual function = {dual_list_fw[-1]:.3e}")
            print(f"Number of non-zero components of x = {active_set_size_list_fw[-1]}")
            print(f"Number of iterations = {num_iterations_fw}")
            print(f"Total CPU time: {CPU_time_list_fw[-1]}")
            print(f"center: {center_awayStep_FW} and radius: {radius_awayStep_FW} ")

            asfw = {
                "center": center_awayStep_FW.tolist(),
                "radius": radius_awayStep_FW.tolist(),
                "Last_active_set_size": int(active_set_size_list_fw[-1]),
                "Number of iterations": num_iterations_fw,
                "dual_function_list": [float(value) for value in dual_list_fw],
                "dual_gap_list": [float(value) for value in dual_gap_list_fw],
                "CPU_time": CPU_time_list_fw,
            }

            iterations_away = list(range(num_iterations_fw))

            # Plot graphs for awayStep_FW
            graph_path = os.path.join(incremented_path, "asfw_graphs")
            os.mkdir(graph_path)
            plot_points_circle(A, radius_awayStep_FW, center_awayStep_FW, title, graph_path, show_graphs)
            plot_cpu_time_vs_dual_gap(CPU_time_list_fw, dual_list_fw, title, graph_path, show_graphs)
            plot_active_set_size_vs_dual_gap(active_set_size_list_fw, dual_gap_list_fw, title, graph_path,
                                             show_graphs)
            plot_cpu_time_vs_objective_function(CPU_time_list_fw, dual_list_fw, title, graph_path, show_graphs)
            plot_iterations_vs_objective_function(iterations_away, dual_list_fw, title, graph_path, show_graphs)
            plot_dual_gap_vs_iterations(iterations_away, dual_gap_list_fw, title, graph_path, show_graphs)

        if method == "bpfw":
            print("\n*****************")
            title = "*  Blended Pairwise FW   *"
            print(title)
            print("*****************")

            logging.info("\nBPFW algorithm started!")
            (center_bpfw,
             radius_bpfw,
             active_set_size_list_bpfw,
             dual_gap_list_bpfw,
             dual_list_bpfw,
             CPU_time_list_bpfw) = blendedPairwise_FW(A, epsilon, maxiter)
            # TODO: Debug - Here for dual_bp sometimes we get positive value, sometimes negative.

            # Print results:
            num_iterations_bpfw = len(dual_list_bpfw)
            print(f"dual function = {dual_list_bpfw[-1]:.3e}")
            print(f"Number of non-zero components of x = {active_set_size_list_bpfw[-1]}")
            print(f"Number of iterations = {num_iterations_bpfw}")
            print(f"Total CPU time: {CPU_time_list_bpfw[-1]}")
            print(f"center: {center_bpfw} and radius: {radius_bpfw} ")

            bpfw = {
                "center": center_bpfw.tolist(),
                "radius": radius_bpfw.tolist(),
                "Last_active_set_size": int(active_set_size_list_bpfw[-1]),
                "Number of iterations": num_iterations_bpfw,
                "dual_function_list": [float(value) for value in dual_list_bpfw],
                "dual_gap_list": [float(value) for value in dual_gap_list_bpfw],
                "CPU_time": CPU_time_list_bpfw,
            }

            iterations_blended = list(range(num_iterations_bpfw))

            # Plot graphs for blendedPairwise_FW
            graph_path = os.path.join(incremented_path, "bpfw_graphs")
            os.mkdir(graph_path)
            plot_points_circle(A, radius_bpfw, center_bpfw, title, graph_path, show_graphs)
            plot_cpu_time_vs_dual_gap(CPU_time_list_bpfw, dual_gap_list_bpfw, title, graph_path, show_graphs)
            plot_active_set_size_vs_dual_gap(active_set_size_list_bpfw, dual_gap_list_bpfw, title, graph_path,
                                             show_graphs)
            plot_cpu_time_vs_objective_function(CPU_time_list_bpfw, dual_list_bpfw, title, graph_path,
                                                show_graphs)
            plot_iterations_vs_objective_function(iterations_blended, dual_list_bpfw, title, graph_path,
                                                  show_graphs)
            plot_dual_gap_vs_iterations(iterations_blended, dual_gap_list_bpfw, title, graph_path, show_graphs)

        if method == "appfw":
            print("\n*****************")
            title = "*  (1+epsilon)-approximation FW   *"
            print(title)
            print("*****************")
            logging.info("\n(1+epsilon)-approximation FW algorithm started!")
            (center_aproxAlg,
             radius_aproxAlg,
             active_set_size_list_aproxAlg,
             dual_gap_list_aproxAlg,
             dual_list_aproxAlg,
             CPU_time_list_aproxAlg) = one_plus_eps_MEB_approximation(A, epsilon, maxiter)

            # Print results:
            num_iterations_aproxAlg = len(dual_list_aproxAlg)
            print(f"dual function = {dual_list_aproxAlg[-1]:.3e}")
            print(f"Number of non-zero components of x = {active_set_size_list_aproxAlg[-1]}")
            print(f"Number of iterations = {num_iterations_aproxAlg}")
            print(f"Total CPU time: {CPU_time_list_aproxAlg[-1]}")
            print(f"center: {center_aproxAlg} and radius: {radius_aproxAlg} ")

            appfw = {
                "center": center_aproxAlg.tolist(),
                "radius": radius_aproxAlg.tolist(),
                "Last_active_set_size": int(active_set_size_list_aproxAlg[-1]),
                "Number of iterations": num_iterations_aproxAlg,
                "dual_function_list": [float(value) for value in dual_list_aproxAlg],
                "dual_gap_list": [float(value) for value in dual_gap_list_aproxAlg],
                "CPU_time": CPU_time_list_aproxAlg,
            }

            # Plot graphs for one_plus_eps_MEB_approximation
            iterations_approx = list(range(num_iterations_aproxAlg))
            graph_path = os.path.join(incremented_path, "appfw_graphs")
            os.mkdir(graph_path)
            plot_points_circle(A, radius_aproxAlg, center_aproxAlg, title, graph_path, show_graphs)
            plot_cpu_time_vs_dual_gap(CPU_time_list_aproxAlg, dual_gap_list_aproxAlg, title, graph_path,
                                      show_graphs)
            plot_active_set_size_vs_dual_gap(active_set_size_list_aproxAlg, dual_gap_list_aproxAlg, title,
                                             graph_path, show_graphs)
            plot_cpu_time_vs_objective_function(CPU_time_list_aproxAlg, dual_list_aproxAlg, title, graph_path,
                                                show_graphs)
            plot_iterations_vs_objective_function(iterations_approx, dual_list_aproxAlg, title, graph_path,
                                                  show_graphs)
            plot_dual_gap_vs_iterations(iterations_approx, dual_gap_list_aproxAlg, title, graph_path, show_graphs)

    # Create yaml content
    output = {
        'config': config,
        'asfw': asfw,
        'bpfw': bpfw,
        'appfw': appfw,
    }

    # Save output yaml file
    with open(os.path.join(incremented_path, 'output.yaml'), 'w') as file:
        yaml.dump(output, file)
        logging.info(f"Output.yaml created")
