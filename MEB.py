import numpy as np
from src.logger import logging
from src.Frank_Wolfe import FW_Q

if __name__ == '__main__':


    logging.info("Creating data points")
    # Number of samples
    m = 2 ** 5

    # Number of variables
    n = 2 ** 12

    # Max number of iteartions
    maxit = 3000

    # initial solution
    u0 = np.zeros(n)
    u0[0] = 1e0

    A = np.random.randn(m, n)

    eps = 1e-6

    print("*****************")
    print("*  FW STANDARD  *")
    print("*****************")

    logging.info("Frank Wolfe algorithm started!")
    u_fw, iter_fw, dual_fw = FW_Q(A, u0, eps, maxit)

    # Print results:
    print(f"dual function = {dual_fw:.3e}")
    print(f"Number of non-zero components of x = {np.sum(np.abs(u_fw) >= 0.0001)}")
    print(f"Number of iterations = {iter_fw}")