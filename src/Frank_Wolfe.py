import numpy as np
import time
from src.logger import logging

def FW_Q(A, u, eps,maxit):

    _, n = A.shape

    A2 = A.T @ A
    sum_a2 = np.sum(A ** 2, axis=0)
    dual_func = u.T @ A2 @ u - sum_a2.T @ u

    k = 1 # iterations
    logging.info(f"First value of dual function {dual_func:.3e}")
    logging.info("Frank Wolfe algorithm first iteration started!")

    tstart = time.time()

    while k <= maxit:



        A2u = A2 @ u  #calculate A**2 @ u just once here to make algorithm faster, because we are gonna use it twice

        # objective function
        dual_func = u.T @ A2u - sum_a2.T @ u

        # gradient evaluation
        grad = 2 * A2u - sum_a2

        # solution of FW problem
        i_star = np.argmin(grad)
        u_star = np.zeros(n)
        u_star[i_star] = 1.0

        # direction calculation
        Fwd = u_star - u  # Frank Wolfe direction
        gnr = - grad.T @ Fwd

        # stopping criteria
        if gnr <= eps:
            break

        # Fixed learning rate
        alpha = 2.0 / (k + 1)

        # update variables
        u = u + alpha * Fwd



        k += 1
    ttot = time.time() - tstart

    logging.info("Frank Wolfe algorithm iterations finished!")
    logging.info(f"Last value of dual function {dual_func:.3e}")
    logging.info(f"Total CPU time {ttot:.3e}")

    return u, k, dual_func


