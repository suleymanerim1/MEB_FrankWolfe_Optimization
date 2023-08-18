import numpy as np
from src.logger import logging
from src.Frank_Wolfe import awayStep_FW, blendedPairwise_FW

if __name__ == '__main__':


    logging.info("Creating data points")
    # Number of samples
    m = 2 ** 5

    # Dimension of variables
    n = 2 ** 4

    # Max number of iteartions
    maxit = 50000

    # initial solution
    u0 = np.zeros(m)
    u0[0] = 1e0

    A = np.random.randn(n, m)

    eps = 1e-6

    print("*****************")
    print("*  Away Step FW   *")
    print("*****************")

    logging.info("\nASFW algorithm started!")
    u_fw, iter_fw, dual_fw, CPU_time_fw = awayStep_FW(A, u0, eps, maxit)
    radius_awayStep_FW = np.sqrt(-dual_fw)
    a_sum = np.sum(A,axis=0)
    center_awayStep_FW = a_sum * u_fw

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
    u_bp, iter_bp, dual_bp, CPU_time_bp = blendedPairwise_FW(A, u0, eps, maxit)
    radius_awayStep_BP = np.sqrt(-dual_bp)
    a_sum = np.sum(A,axis=0)
    center_awayStep_BP = a_sum * u_bp

    # Print results:
    print(f"dual function = {dual_bp:.3e}")
    print(f"Number of non-zero components of x = {np.sum(np.abs(dual_bp) >= 0.0001)}")
    print(f"Number of iterations = {iter_bp}")
    print(f"CPU time: {CPU_time_bp}")
    print(f"center: {center_awayStep_BP} and radius: {radius_awayStep_BP} ")

    logging.info("BP Frank Wolfe finished!")
    logging.info(f"center: {center_awayStep_BP} and radius: {radius_awayStep_BP} ")
