import numpy as np
from src.logger import logging
from src.FrankWolfeVariants import awayStep_FW, blendedPairwise_FW, one_plus_eps_MEB_approximation
from src.utils import generateRandomMatrix, plot_cpu_time_vs_dual_gap, plot_active_set_size_vs_dual_gap,\
    plot_cpu_time_vs_objective_function, plot_iterations_vs_objective_function, plot_dual_gap_vs_iterations
# from src.utils import fermat_spiral

if __name__ == '__main__':

    maxiter = 1000
    epsilon = 1e-6

    logging.info("Creating data points")
    m = 2 ** 5  # Number of samples
    n = 2 ** 1  # Dimension of variables
    A = generateRandomMatrix(n, m)
    # A = fermat_spiral(m).T

    print("*****************")
    print("*  Away Step FW   *")
    print("*****************")

    logging.info("\nASFW algorithm started!")
    u_fw, iter_fw, dual_fw, CPU_time_fw = awayStep_FW(A, epsilon, maxiter)
    radius_awayStep_FW = np.sqrt(-dual_fw)
    center_awayStep_FW = A @ u_fw

    # Print results:
    print(f"dual function = {dual_fw:.3e}")
    print(f"Number of non-zero components of x = {np.sum(np.abs(u_fw) >= 0.0001)}")
    print(f"Number of iterations = {iter_fw}")
    print(f"CPU time: {CPU_time_fw}")
    print(f"center: {center_awayStep_FW} and radius: {radius_awayStep_FW} ")

    logging.info("Away step Frank Wolfe finished!")
    logging.info(f"center: {center_awayStep_FW} and radius: {radius_awayStep_FW} ")

    print("*****************")
    print("*  Blended Pairwise FW   *")
    print("*****************")

    logging.info("\nBPFW algorithm started!")
    u_bp, iter_bp, dual_bp, CPU_time_bp = blendedPairwise_FW(A, epsilon, maxiter)
    # TODO: Debug - Here for dual_bp sometimes we get positive value, sometimes negative.
    radius_awayStep_BP = np.sqrt(abs(dual_bp))
    center_awayStep_BP = A @ u_bp

    # Print results:
    print(f"dual function = {dual_bp:.3e}")
    print(f"Number of non-zero components of x = {np.sum(np.abs(dual_bp) >= 0.0001)}")
    print(f"Number of iterations = {iter_bp}")
    print(f"CPU time: {CPU_time_bp}")
    print(f"center: {center_awayStep_BP} and radius: {radius_awayStep_BP} ")

    logging.info("BP Frank Wolfe finished!")
    logging.info(f"center: {center_awayStep_BP} and radius: {radius_awayStep_BP} ")

    print("*****************")
    print("*  (1+epsilon)-approximation FW   *")
    print("*****************")
    center_aproxAlg, iteration_aproxAlg, Xk_active_set, u_dual_sol_aproxAlg, radius_aproxAlg, total_time_aproxAlg = \
        one_plus_eps_MEB_approximation(A, epsilon, maxiter)
    # Print results:
    print(f"dual function = {u_dual_sol_aproxAlg}")
    print(f"Number of non-zero components of x = {np.sum(np.abs(u_dual_sol_aproxAlg) >= 0.0001)}")
    print(f"Number of iterations = {iteration_aproxAlg}")
    print(f"CPU time: {total_time_aproxAlg}")
    print(f"center: {center_aproxAlg} and radius: {radius_aproxAlg} ")

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