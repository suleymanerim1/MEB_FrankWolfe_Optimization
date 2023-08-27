import numpy as np
import yaml
import os
from src.logger import my_logger
from src.FrankWolfeVariants import awayStep_FW, blendedPairwise_FW, one_plus_eps_MEB_approximation
from src.utils import increment_path,load_config,generateRandomMatrix, plot_points_circle, fermat_spiral
# from src.utils import plot_cpu_time_vs_dual_gap, plot_active_set_size_vs_dual_gap, plot_cpu_time_vs_objective_function,\
#   plot_iterations_vs_objective_function, plot_dual_gap_vs_iterations, fermat_spiral

# Change only this
yaml_name = "exp2.yaml"



# folder to load config file
config_path = "configs/"
# hyperparameters
config = load_config(yaml_name,config_path)
maxiter = eval(config.get('maxiter'))
epsilon = eval(config.get('epsilon'))
m = eval(config.get('number_of_samples'))
n = eval(config.get('number_of_variables'))
solver_methods = config.get('solver_methods')
data_creation_method = config.get('data_creation_method')


#TODO: Suleyman's question
# What is support vector in graph?

# TODO: return cpu time as cumulative time list -- for Süleyman
# TODO: solve MEB problem and find anomaly --points for Suleyman
# TODO: keep track of anomaly points --for Suleyman
# TODO: important --- 3rd algorithm always stops at first iteration eps condition, check why?? -- for Suleyman
# TODO: 2dn algoritm return negative and positive, check that it creates error
# TODO: save graphs in png -- for Marija
# TODO: find 2 datasets to check -- for Marija


if __name__ == '__main__':

    # Save path
    base_path = 'runs/'
    experiment_path = os.path.join(base_path, os.path.splitext(yaml_name)[0])
    # if there is an experiment with same experiment.yaml, increment_path_number exp1, exp2....
    incremented_path = increment_path(experiment_path, exist_ok=False, sep='_', mkdir=True)
    print(f"Results will be saved: {incremented_path}")

    logging = my_logger(incremented_path)
    #maxiter = 1000
    #epsilon = 1e-6

    #m = 2 ** 10  # Number of samples
    #n = 2 ** 4  # Dimension of variables

    #methods = ["asfw", "bpfw", "appfw"]
    #methods = ["appfw"]


    logging.info("Creating data points")
    if data_creation_method == "random":
        A = generateRandomMatrix(n, m)
    elif data_creation_method == "fermat":
        A = fermat_spiral(m).T
    else:
        pass # TODO: choose 2 datasets and add in this method

    for method in solver_methods:
        # initialize output dictionaries
        aspw={}
        bpfw={}
        appfw={}
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
            print(f"dual function = {dual_list_fw[-1]:.3e}")
            print(f"Number of non-zero components of x = {active_set_size_list_fw[-1]}")
            print(f"Number of iterations = {len(dual_list_fw)}")
            print(f"CPU time: {CPU_time_list_fw}")
            print(f"center: {center_awayStep_FW} and radius: {radius_awayStep_FW} ")

            asfw = {
                "center":center_awayStep_FW.tolist(),
                "radius":radius_awayStep_FW.tolist(),
                "Last_active_set_size": int(active_set_size_list_fw[-1]),
                "Number of iterations" : len(dual_list_fw),
                "dual_function_list": [float(value) for value in dual_list_fw],
                "dual_gap_list": [float(value) for value in dual_gap_list_fw],
                "CPU_time":CPU_time_list_fw,
            }

            plot_points_circle(A, radius_awayStep_FW, center_awayStep_FW, title)

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
            print(f"dual function = {dual_list_bpfw[-1]:.3e}")
            print(f"Number of non-zero components of x = {active_set_size_list_bpfw[-1]}")
            print(f"Number of iterations = {len(dual_list_bpfw)}")
            print(f"CPU time: {CPU_time_list_bpfw}")
            print(f"center: {center_bpfw} and radius: {radius_bpfw} ")
            plot_points_circle(A, radius_bpfw, center_bpfw, title)

            bpfw = {
                "center":center_bpfw.tolist(),
                "radius":radius_bpfw.tolist(),
                "Last_active_set_size": int(active_set_size_list_bpfw[-1]),
                "Number of iterations" : len(dual_list_bpfw),
                "dual_function_list": [float(value) for value in dual_list_bpfw],
                "dual_gap_list": [float(value) for value in dual_gap_list_bpfw],
                "CPU_time":CPU_time_list_bpfw,
            }

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
            print(f"dual function = {dual_list_aproxAlg[-1]:.3e}")
            print(f"Number of non-zero components of x = {active_set_size_list_aproxAlg[-1]}")
            print(f"Number of iterations = {len(dual_list_aproxAlg)}")
            print(f"CPU time: {CPU_time_list_aproxAlg}")
            print(f"center: {center_aproxAlg} and radius: {radius_aproxAlg} ")
            plot_points_circle(A, radius_aproxAlg, center_aproxAlg, title)

            appfw = {
                "center":center_aproxAlg.tolist(),
                "radius":radius_aproxAlg.tolist(),
                "Last_active_set_size": int(active_set_size_list_aproxAlg[-1]),
                "Number of iterations" : len(dual_list_aproxAlg),
                "dual_function_list": [float(value) for value in dual_list_aproxAlg],
                "dual_gap_list": [float(value) for value in dual_gap_list_aproxAlg],
                "CPU_time":CPU_time_list_aproxAlg,
            }

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




    # For testing: (Added by Marija)
    # dual_gap_values_away = dual_gap_list_fw # List of dual gap values for awayStep_FW
    # cpu_time_away = CPU_time_list_fw # List of CPU time for awayStep_FW
    # active_set_sizes_away = active_set_size_list_fw # List of active set sizes for awayStep_FW
    # objective_function_values_away = dual_list_fw  # List of objective function values for awayStep_FW
    # num_iterations_away = {len(dual_list_fw)}
    # iterations_away = list(range(num_iterations_away + 1))
    #
    # dual_gap_values_blended = dual_gap_list_bpfw # List of dual gap values for blendedPairwise_FW
    # cpu_time_blended = CPU_time_list_bpfw # List of CPU time for blendedPairwise_FW
    # active_set_sizes_blended = active_set_size_list_bpfw # List of active set sizes for blendedPairwise_FW
    # objective_function_values_blended = dual_list_bpfw # List of objective function values for blendedPairwise_FW
    # num_iterations_blended = {len(dual_list_bpfw)}
    # iterations_blended = list(range(num_iterations_blended + 1))
    #
    # dual_gap_values_approx = dual_gap_list_aproxAlg # List of dual gap values for blendedPairwise_FW
    # cpu_time_approx = CPU_time_list_aproxAlg # List of CPU time for blendedPairwise_FW
    # active_set_sizes_approx = active_set_size_list_aproxAlg # List of active set sizes for blendedPairwise_FW
    # objective_function_values_approx = dual_list_aproxAlg # List of objective function values for blendedPairwise_FW
    # num_iterations_approx = {len(dual_list_aproxAlg)}
    # iterations_approx = list(range(num_iterations_approx + 1))
    #
    # path = incremented_path
    # show = True
    #
    # # Plot graphs for awayStep_FW
    # plot_cpu_time_vs_dual_gap(cpu_time_away, dual_gap_values_away, 'awayStep_FW', path, show)
    # plot_active_set_size_vs_dual_gap(active_set_sizes_away, dual_gap_values_away, 'awayStep_FW', path, show)
    # plot_cpu_time_vs_objective_function(cpu_time_away, objective_function_values_away, 'awayStep_FW', path ,show)
    # plot_iterations_vs_objective_function(iterations_away, objective_function_values_away, 'awayStep_FW', path, show)
    # plot_dual_gap_vs_iterations(iterations_away, dual_gap_values_away, 'awayStep_FW', path, show)
    #
    # # Plot graphs for blendedPairwise_FW
    # plot_cpu_time_vs_dual_gap(cpu_time_blended, dual_gap_values_blended, 'blendedPairwise_FW', path, show)
    # plot_active_set_size_vs_dual_gap(active_set_sizes_blended, dual_gap_values_blended, 'blendedPairwise_FW', path, show)
    # plot_cpu_time_vs_objective_function(cpu_time_blended, objective_function_values_blended, 'blendedPairwise_FW', path, show)
    # plot_iterations_vs_objective_function(iterations_blended, objective_function_values_blended, 'blendedPairwise_FW', path, show)
    # plot_dual_gap_vs_iterations(iterations_blended, dual_gap_values_blended, 'blendedPairwise_FW', path, show)
    #
    # # Plot graphs for one_plus_eps_MEB_approximation
    # plot_cpu_time_vs_dual_gap(cpu_time_approx, dual_gap_values_approx, 'one_plus_eps_MEB_approximation', path, show)
    # plot_active_set_size_vs_dual_gap(active_set_sizes_approx, dual_gap_values_approx,'one_plus_eps_MEB_approximation', path, show)
    # plot_cpu_time_vs_objective_function(cpu_time_approx, objective_function_values_approx,'one_plus_eps_MEB_approximation', path, show)
    # plot_iterations_vs_objective_function(iterations_approx, objective_function_values_approx,'one_plus_eps_MEB_approximation', path, show)
    # plot_dual_gap_vs_iterations(iterations_approx, dual_gap_values_approx, 'one_plus_eps_MEB_approximation', path, show)
