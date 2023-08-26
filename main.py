import numpy as np
from src.logger import logging
from src.FrankWolfeVariants import awayStep_FW, blendedPairwise_FW, one_plus_eps_MEB_approximation
from src.utils import generateRandomMatrix, plot_points_circle, fermat_spiral
# from src.utils import plot_cpu_time_vs_dual_gap, plot_active_set_size_vs_dual_gap, plot_cpu_time_vs_objective_function,\
#   plot_iterations_vs_objective_function, plot_dual_gap_vs_iterations, fermat_spiral


#TODO: Suleyman's question
# What is support vector in graph?

# TODO: important --- 3rd algorithm always stops at first iteration eps condition, check why??
# TODO: save results to json
# TODO: save graphs in png
# TODO: find 2 datasets to check
if __name__ == '__main__':

    maxiter = 1000
    epsilon = 1e-6

    logging.info("Creating data points")
    m = 2 ** 10  # Number of samples
    n = 2 ** 4  # Dimension of variables
    #A = generateRandomMatrix(n, m)
    A = fermat_spiral(m).T
    #methods = ["asfw", "bpfw", "appfw"]
    methods = ["appfw"]

    for method in methods:
        if method == "asfw":

            print("*****************")
            title = "*  Away Step FW   *"
            print(title)
            print("*****************")

            logging.info("\nASFW algorithm started!")
            (center_awayStep_FW,
             radius_awayStep_FW,
             active_set_size_list_fw,
             dual_gap_list_fw,
             dual_list_fw,
             CPU_time_fw) = awayStep_FW(A, epsilon, maxiter)

            # Print results:
            print(f"dual function = {dual_list_fw[-1]:.3e}")
            print(f"Number of non-zero components of x = {active_set_size_list_fw[-1]}")
            print(f"Number of iterations = {len(dual_list_fw)}")
            print(f"CPU time: {CPU_time_fw}")
            print(f"center: {center_awayStep_FW} and radius: {radius_awayStep_FW} ")

            plot_points_circle(A, radius_awayStep_FW, center_awayStep_FW, title)

        if method == "bpfw":

            print("*****************")
            title = "*  Blended Pairwise FW   *"
            print(title)
            print("*****************")

            logging.info("\nBPFW algorithm started!")
            (center_bpfw,
             radius_bpfw,
             active_set_size_list_bpfw,
             dual_gap_list_bpfw,
             dual_list_bpfw,
             CPU_time_bpfw) = blendedPairwise_FW(A, epsilon, maxiter)
            # TODO: Debug - Here for dual_bp sometimes we get positive value, sometimes negative.

            # Print results:
            print(f"dual function = {dual_list_bpfw[-1]:.3e}")
            print(f"Number of non-zero components of x = {active_set_size_list_bpfw[-1]}")
            print(f"Number of iterations = {len(dual_list_bpfw)}")
            print(f"CPU time: {CPU_time_bpfw}")
            print(f"center: {center_bpfw} and radius: {radius_bpfw} ")
            plot_points_circle(A, radius_bpfw, center_bpfw, title)

        if method == "appfw":

            print("*****************")
            title = "*  (1+epsilon)-approximation FW   *"
            print(title)
            print("*****************")
            logging.info("\n(1+epsilon)-approximation FW algorithm started!")
            (center_aproxAlg,
             radius_aproxAlg,
             active_set_size_list_aproxAlg,
             dual_gap_list_aproxAlg,
             dual_list_aproxAlg,
             CPU_time_aproxAlg) = one_plus_eps_MEB_approximation(A, epsilon, maxiter)

            # Print results:
            print(f"dual function = {dual_list_aproxAlg[-1]:.3e}")
            print(f"Number of non-zero components of x = {active_set_size_list_aproxAlg[-1]}")
            print(f"Number of iterations = {len(dual_list_aproxAlg)}")
            print(f"CPU time: {CPU_time_aproxAlg}")
            print(f"center: {center_aproxAlg} and radius: {radius_aproxAlg} ")
            plot_points_circle(A, radius_aproxAlg, center_aproxAlg, title)

    # For testing: (Added by Marija)
    # dual_gap_values_away = [...]  # List of dual gap values for awayStep_FW
    # cpu_time_away = [...]  # Value of CPU time for awayStep_FW
    # active_set_sizes_away = [...]  # List of active set sizes for awayStep_FW
    # objective_function_values_away = [...]  # List of objective function values for awayStep_FW
    # iterations_away = list(range(1, len(objective_function_values_away) + 1))  # ??
    #
    # dual_gap_values_blended = [...]  # List of dual gap values for blendedPairwise_FW
    # cpu_time_blended = [...]  # Value of CPU time for blendedPairwise_FW
    # active_set_sizes_blended = [...]  # List of active set sizes for blendedPairwise_FW
    # objective_function_values_blended = [...]  # List of objective function values for blendedPairwise_FW
    # iterations_blended = list(range(1, len(objective_function_values_blended) + 1))  # ??
    #
    # # Plot graphs for awayStep_FW
    # plot_cpu_time_vs_dual_gap(cpu_time_away, dual_gap_values_away, 'awayStep_FW')
    # plot_active_set_size_vs_dual_gap(active_set_sizes_away, dual_gap_values_away, 'awayStep_FW')
    # plot_cpu_time_vs_objective_function(cpu_time_away, objective_function_values_away, 'awayStep_FW')
    # plot_iterations_vs_objective_function(iterations_away, objective_function_values_away, 'awayStep_FW')
    # plot_dual_gap_vs_iterations(iterations_away, dual_gap_values_away, 'awayStep_FW')
    #
    # # Plot graphs for blendedPairwise_FW
    # plot_cpu_time_vs_dual_gap(cpu_time_blended, dual_gap_values_blended, 'blendedPairwise_FW')
    # plot_active_set_size_vs_dual_gap(active_set_sizes_blended, dual_gap_values_blended, 'blendedPairwise_FW')
    # plot_cpu_time_vs_objective_function(cpu_time_blended, objective_function_values_blended, 'blendedPairwise_FW')
    # plot_iterations_vs_objective_function(iterations_blended, objective_function_values_blended, 'blendedPairwise_FW')
    # plot_dual_gap_vs_iterations(iterations_blended, dual_gap_values_blended, 'blendedPairwise_FW')
