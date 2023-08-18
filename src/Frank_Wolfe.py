import numpy as np
import math
import time
from src.logger import logging

# TODO: away step FW rewrite step by step, change variables names like blended Pairwise, use dual_func and gradient and line search
# TODO: away step arrange active set weights by article
# TODO: blended pairwise think about active set weights
# TODO: return radius and center
# TODO: blended does not have stopping condition, why?
# TODO: add last algorithm
# TODO: add stopping condition of time for each alogirthm
# TODO: add max iteration stopping condition for each algorithm if neccesaary
# TODO: min grad stopping condition for each algorithm if neceesary
# TODO: run each algorithm and check for error from logs and results, check how each variables acts according to article
def dual_function(A, u):
    A2 = A.T @ A
    A2u = A2 @ u
    sum_a2 = np.sum(A ** 2, axis=0)
    return u.T @ A2u - sum_a2.T @ u


def gradient(A, u):
    A2 = A.T @ A
    A2u = A2 @ u
    sum_a2 = np.sum(A ** 2, axis=0)

    return 2 * A2u - sum_a2

def golden_section_search(func, a , b, tol=1e-6, max_iter=100):
    """
    Perform exact line search using the golden section search method.

    Parameters:
        func (callable): The objective function to minimize.
        a (float): The left endpoint of the interval.
        b (float): The right endpoint of the interval.
        tol (float): Tolerance for stopping criterion.
        max_iter (int): Maximum number of iterations.

    Returns:
        float: The estimated minimum point along the interval.
    """
    # Calculate the golden ratio
    golden_ratio = (math.sqrt(5) - 1) / 2

    x1 = a + (1 - golden_ratio) * (b - a)
    x2 = a + golden_ratio * (b - a)

    f_x1 = func(x1)
    f_x2 = func(x2)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break

        if f_x1 < f_x2:
            b = x2
            x2 = x1
            f_x2 = f_x1
            x1 = a + (1 - golden_ratio) * (b - a)
            f_x1 = func(x1)
        else:
            a = x1
            x1 = x2
            f_x1 = f_x2
            x2 = a + golden_ratio * (b - a)
            f_x2 = func(x2)

    return (a + b) / 2




def awayStep_FW(A, u, eps, maxit):
    #Lacoste-Julien
    tstart = time.time()


    _, n = A.shape


    alpha_max = 1  # max step_size
    alpha_t = 0  # step_size at iteration t

    # step 1
    St = np.where(u > 0)[0]  # active set St -- initialize


    logging.info("Away Step Frank Wolfe algorithm first iteration started!")

    # step 2
    for t in range(maxit):
        logging.info(f"\n--------------Iteration {t} -----------------")


        # objective function
        dual = dual_function(A, u)
        logging.info(f"Dual func value found: {dual} ")

        # gradient evaluation
        grad = gradient(A,u)
        logging.info(f"Gradient calculation done. ")


        # solution of FW problem - step 3
        s_t_idx = np.argmin(grad) #find index which makes gradient min
        s_t = np.zeros(n)  # create n dimensional array and do e_i = 1
        s_t[s_t_idx] = 1.0
        logging.info(f"FW min grad unit simplex index found: {s_t_idx}")

        # calculate FW direction
        d_FW = s_t - u
        logging.info(f"FW direction calculated.")


        # solution of Away Step - step 4 in Lacoste-Julien algo1
        max_grad_idx = np.argmax(grad[St])  # find max_grad_index from active set St
        v_t_idx = St[max_grad_idx]  # set away vertex index from active set St
        v_t = np.zeros(n)
        v_t[v_t_idx] = 1.0  # create n dimensional array and do a_t = 1 and others 0
        logging.info(f"Away step max grad unit simplex index found: {v_t_idx}")


        # calculate Away Step direction
        d_AW = u - v_t
        logging.info(f"Away Step direction calculated")

        # stopping condition
        # Step 5 - FW gap is small enough, so return
        g_FW = -grad.T @ d_FW
        if g_FW  <= eps:
            logging.info(f"Stopping condition gap < epsilon is met!")
            return u

        # step 6 - compare FW gap and AW gap
        g_AW = -grad.T @ d_AW  # calculate Away step gap
        fw_check = True
        if g_FW >= g_AW:

            #step 7 - set direction at iteration k as FW direction
            d_k = d_FW
            alpha_max = 1  # set max step_size
            logging.info(f"Choose Frank Wolfe direction, step_size_max: {alpha_max}")


        # step 8
        else:

            #step 9 - choose away direction and max feasible step_size
            d_k = d_AW
            alpha_max = St[v_t_idx] / (1-St[v_t_idx]) # use the value of unit simplex highest grad coordinate
            logging.info(f"Choose Away Step  direction, step_size_max : {alpha_max}")
            fw_check = False

        #step 10 end if
        # step 11- a line search
        alpha_t = golden_section_search(dual(A, u), a=0, b=alpha_max)

        logging.info(f"Step size is set {alpha_t}")

        #step 12
        u = u + alpha_t * d_k
        logging.info(f"u values updated")

        # step 13 update active set (Lacoste-Juline pg.4 1st paragraph)
        if fw_check:  # fw step used
                St= (1-alpha_t)*St
                St[s_t_idx] = St[s_t_idx]+alpha_t


        else:  # away step used
            St = (1+alpha_t) * St
            St[v_t_idx] = St[v_t_idx] - alpha_t

        logging.info(f"Active set is updated!:{St}")
        # step 14 end for



    ttot = time.time() - tstart

    logging.info("Frank Wolfe algorithm iterations finished!")
    logging.info(f"Last value of dual function {dual:.3e}")
    logging.info(f"Total CPU time {ttot:.3e}")

    return u, t, dual


def blendedPairwise_FW(A, u, eps, maxit):

    # Tsuji Algorithm 1
    tstart = time.time()

    _, n = A.shape

    alpha_max = 1  # max step_size
    alpha_t = 0  # step_size at iteration t


    # step 1
    St = np.where(u > 0)[0]  # active set St -- initialize

    logging.info("Blended Pairwise Frank Wolfe algorithm first iteration started!")
    # step 2
    for t in range(maxit):
        logging.info(f"\n--------------Iteration {t} -----------------")


        # objective function
        dual = dual_function(A, u)
        logging.info(f"Dual func value found: {dual} ")

        # gradient evaluation
        grad = gradient(A,u)
        logging.info(f"Gradient calculation done. ")

        # step 3 - away vertex
        max_grad_idx = np.argmax(grad[St])  # find max_grad_index from active set St
        a_t_idx = St[max_grad_idx]  # set away vertex index from active set St
        a_t = np.zeros(n)
        a_t[a_t_idx] = 1.0  # create n dimensional array and do a_t = 1 and others 0
        logging.info(f"Away step max grad unit simplex index found: {a_t_idx}")

        # step 4 - local FW
        min_grad_idx = np.argmin(grad[St])  # find min_grad_index from active set St
        s_t_idx = St[min_grad_idx]  # set local FW index from active set St
        s_t = np.zeros(n)
        s_t[s_t_idx] = 1.0  # create n dimensional array and do s_t = 1 and others 0
        logging.info(f"Away step max grad unit simplex index found: {s_t_idx}")

        # step 5 - global FW
        w_t_idx = np.argmin(grad)  # find index which makes gradient min - search in all vertices
        w_t = np.zeros(n)
        w_t[w_t_idx] = 1.0  # create n dimensional array and do w_t = 1 and others 0
        logging.info(f"FW min grad unit simplex index found: {w_t_idx}")

        # step 6
        if grad.T @ (a_t-s_t) >= grad.T @ (u - w_t):

            # step 7
            d_t = a_t - s_t  # away_vertex - local FW

            # step 8
            alpha_max = u[a_t_idx]

            # step 9
            alpha_t = alpha_t  = golden_section_search(dual(A,u), a =0 , b = alpha_max)
            # step 10
            if alpha_t < alpha_max:
                # step 11
                St = St  # descent step, do not update St

            # step 12
            else:
                # step 13
                St[a_t_idx] = 0  # drop step
        # step 14 - end if
        # step 15
        else:  # FW Step
            # step 16
            d_t = u - w_t  # u weights - global FW

            # step 17
            alpha_t  = golden_section_search(dual(A,u), a =0 , b = 1)
            # step 18
            if alpha_t == 1:
                St = w_t
            else:
                St[w_t_idx] = 1

        # step 19 - end if
        # step 20
        u = u - alpha_t * d_t
    # step 21 - end for

    ttot = time.time() - tstart

    logging.info("Frank Wolfe algorithm iterations finished!")
    logging.info(f"Last value of dual function {dual:.3e}")
    logging.info(f"Total CPU time {ttot:.3e}")

    return u, t, dual


