import numpy as np
import math
import time
from src.logger import logging

def compute_A_related_variables(A, u):
    A_squared = A.T @ A
    A_squared_u = A_squared @ u
    sum_A_squared = np.sum(A_squared, axis=0)
    return A_squared, A_squared_u, sum_A_squared

def dual_function(A, u):
    _, A_squared_u, sum_A_squared = compute_A_related_variables(A, u)
    return u.T @ A_squared_u - sum_A_squared.T @ u

def gradient(A, u):
    _, A_squared_u, sum_A_squared = compute_A_related_variables(A, u)
    return 2 * A_squared_u - sum_A_squared

def golden_section_search(func, A, u, d_t, a, b,
                          tol=1e-6, max_iter=100):
    """
    Perform exact line search using the golden section search method.

    Parameters:
        func (callable): The objective function to minimize.
        A (matrix): points matrix (m,n)
        u (vector) : convex combination weights for MEB dual problem
        d_t (vector) : search direction for line search
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

    u1 = u + d_t * x1
    u2 = u + d_t * x2
    dual1 = func(A, u1)
    dual2 = func(A, u2)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break

        if dual1 < dual2:
            b = x2
            x2 = x1
            dual2 = dual1
            x1 = a + (1 - golden_ratio) * (b - a)
            u1 = u + d_t * x1
            dual1 = func(A, u1)
        else:
            a = x1
            x1 = x2
            dual1 = dual2
            x2 = a + golden_ratio * (b - a)
            u2 = u + d_t * x2
            dual2 = func(A, u2)

    return (a + b) / 2

def awayStep_FW(A, eps, max_iter, line_search_strategy='golden_search'):
    # Lacoste-Julien
    logging.info("Away Step Frank Wolfe algorithm first iteration started!")

    #initialize output lists
    active_set_size_list = [1]
    dual_gap_list = [0]
    dual_list = [0]
    total_time = 0
    CPU_time_list = [0]

    n, m = A.shape
    logging.info(f"Dataset size: {m} points, each {n}-dimensional.")
    # alpha_max = 1  # max step_size
    # alpha_t = 0  # step_size at iteration t
    dual = 0
    iteration = 0
    # initial solution
    u = np.zeros(m)
    u[0] = 1e0

    # Step 1
    St = np.zeros(m)
    St[np.where(u > 0)[0]] = 1  # active set St -- initialize

    # Step 2
    for iteration in range(max_iter):
        t_start = time.time()
        logging.info(f"\n--------------Iteration {iteration} -----------------")

        # objective function
        dual = dual_function(A, u)
        logging.info(f"Dual function value found: {dual} ")
        dual_list.append(dual)
        dual_gap = dual_list[-1]-dual_list[-2]
        logging.info(f"Dual gap value found: {dual_gap} ")
        dual_gap_list.append(dual_gap)

        # gradient evaluation
        grad = gradient(A, u)

        # solution of FW problem - Step 3
        s_t_idx = np.argmin(grad)  # find index which makes gradient min
        s_t = np.zeros(m)  # create m-dimensional array and set e_i = 1
        s_t[s_t_idx] = 1.0
        logging.info(f"FW min grad unit simplex index found: {s_t_idx}")

        # calculate FW direction
        d_FW = s_t - u

        # solution of Away Step - Step 4 in Lacoste-Julien algo1
        active_idxs_list = np.where(St > 0)[0]
        max_grad_idx = np.argmax(grad[active_idxs_list])  # find max_grad_index from active set St
        v_t_idx = active_idxs_list[max_grad_idx]  # set away vertex index from active set St
        v_t = np.zeros(m)  # create m-dimensional array and set v_t = 1 and others to 0
        v_t[v_t_idx] = 1.0
        logging.info(f"Away step max grad unit simplex index found: {v_t_idx}")

        # calculate Away Step direction
        d_AW = u - v_t

        # Step 5 - If FW gap is small enough, break
        g_FW = -grad.T @ d_FW
        if g_FW  <= eps:  # stopping condition
            logging.info(f"Stopping condition gap < epsilon is met!")
            it_time = time.time() - t_start
            total_time = total_time + it_time
            CPU_time_list.append(total_time)
            active_set_size_list.append(np.sum(np.abs(St) >= 0.0001))

            break
        fw_check = True

        # Step 6 - compare FW gap and AW gap
        g_AW = -grad.T @ d_AW  # calculate Away step gap
        if g_FW >= g_AW:

            # Step 7 - set direction at iteration k as FW direction
            d_t = d_FW
            alpha_max = 1  # set max step_size
            logging.info(f"Choose Frank Wolfe direction, step_size_max: {alpha_max}")

        # Step 8
        else:
            # step 9 - choose away direction and max feasible step_size
            d_t = d_AW
            alpha_max = St[v_t_idx] / (1 - St[v_t_idx])  # use the value of unit simplex highest grad coordinate
            logging.info(f"Choose Away Step  direction, step_size_max : {alpha_max}")
            fw_check = False

        # Step 10 end if

        # Step 11 - Calculate step size using line search
        if line_search_strategy == 'line_search_strategy':
            alpha_t = golden_section_search(dual_function, A, u, d_t, a=0, b=alpha_max)

        else:
            alpha_t = golden_section_search(dual_function, A, u, d_t, a=0, b=alpha_max)

        logging.info(f"Step size is set. --> alpha_t: {alpha_t}")

        # Step 12
        u = u + alpha_t * d_t

        # Step 13 update active set (Lacoste-Juline pg.4 1st paragraph)
        if fw_check:  # fw step used
            St = (1-alpha_t)*St
            St[s_t_idx] = St[s_t_idx]+alpha_t

        else:  # away step used
            St = (1+alpha_t) * St
            St[v_t_idx] = St[v_t_idx] - alpha_t
        active_set_size_list.append(np.sum(np.abs(St) >= 0.0001))
        # Step 14 end for

        it_time = time.time() - t_start
        total_time = total_time + it_time
        CPU_time_list.append(total_time)



    radius = np.sqrt(-dual)
    center = A @ u

    logging.info("\n-----------Away step Frank Wolfe finished!--------------")
    logging.info(f"center: {center} and radius: {radius} ")
    logging.info(f"Last value of dual function {dual:.3e}")
    logging.info(f"Last value of dual gap  {dual_gap:.3e}")
    logging.info(f"Total CPU time {total_time:.3e}")
    logging.info(f"Number of non-zero components of x = {active_set_size_list[-1]}")
    logging.info(f"Number of iterations = {len(dual_list)}")

    return center,radius, active_set_size_list ,dual_gap_list, dual_list, CPU_time_list

def blendedPairwise_FW(A, eps, max_iter=1000):  # Tsuji Algorithm 1
    logging.info("Blended Pairwise Frank Wolfe algorithm first iteration started!")

    #initialize output lists
    active_set_size_list = [1]
    dual_gap_list = [0]
    dual_list = [0]
    total_time = 0
    CPU_time_list = [0]

    n, m = A.shape
    # alpha_max = 1  # max step_size
    # alpha_t = 0  # step_size at iteration t
    logging.info(f"Dataset size: {m} points, each {n}-dimensional.")

    dual = 0
    iteration = 0
    # initial solution
    u = np.zeros(m)
    u[0] = 1e0

    # Step 1
    St = np.zeros(m)
    St[np.where(u > 0)[0]] = 1  # active set St -- initialize

    # Step 2
    for iteration in range(max_iter):
        t_start = time.time()
        logging.info(f"\n--------------Iteration {iteration} -----------------")

        # objective function
        dual = dual_function(A, u)
        logging.info(f"Dual function value found: {dual} ")
        dual_list.append(dual)
        dual_gap = dual_list[-1]-dual_list[-2]
        logging.info(f"Dual gap value found: {dual_gap} ")
        dual_gap_list.append(dual_gap)

        # gradient evaluation
        grad = gradient(A, u)

        # Step 3 - away vertex
        active_idxs_list = np.where(St > 0)[0]
        max_grad_idx = np.argmax(grad[active_idxs_list])  # find max_grad_index from active set St
        a_t_idx = active_idxs_list[max_grad_idx]  # set away vertex index from active set St
        a_t = np.zeros(m)  # create m-dimensional array and set a_t = 1 and others to 0
        a_t[a_t_idx] = 1.0
        logging.info(f"Away step max grad unit simplex index found: {a_t_idx}")

        # Step 4 - local FW
        min_grad_idx = np.argmin(grad[active_idxs_list])  # find min_grad_index from active set St
        s_t_idx = active_idxs_list[min_grad_idx]  # set local FW index from active set St
        s_t = np.zeros(m)  # create m-dimensional array and set s_t = 1 and others to 0
        s_t[s_t_idx] = 1.0
        logging.info(f"Local FW min grad unit simplex index found: {s_t_idx}")

        # Step 5 - global FW
        w_t_idx = np.argmin(grad)  # find index which makes gradient min - search in all vertices
        w_t = np.zeros(m)  # create m-dimensional array and set w_t = 1 and others to 0
        w_t[w_t_idx] = 1.0
        logging.info(f"Global FW min grad unit simplex index found: {w_t_idx}")

        FW_gap = grad.T @ (u - w_t)
        if FW_gap  <= eps:  # stopping condition
            logging.info(f"Stopping condition gap < epsilon is met!")
            it_time = time.time() - t_start
            total_time = total_time + it_time
            CPU_time_list.append(total_time)
            active_set_size_list.append(np.sum(np.abs(St) >= 0.0001))
            break

        # Step 6
        if grad.T @ (a_t-s_t) >= FW_gap:
            # Step 7
            d_t = a_t - s_t  # away_vertex - local FW
            # Step 8
            alpha_max = u[a_t_idx]
            # Step 9
            alpha_t  = golden_section_search(dual_function, A, u, d_t, a=0, b=alpha_max)
            # Step 10
            if alpha_t < alpha_max:
                # Step 11
                St = St  # descent step, do not update St
                logging.info(f"Descent step taken, step_size: {alpha_t}")
            # Step 12
            else:
                # Step 13
                St[a_t_idx] = 0  # drop step
                logging.info(f"Drop step taken, step_size: {alpha_t}")
        # Step 14 - end if

        # Step 15
        else:  # FW Step
            # Step 16
            d_t = u - w_t  # u weights - global FW
            # Step 17
            alpha_t  = golden_section_search(dual_function, A, u, d_t, a=0, b=1)
            logging.info(f"Choose Frank Wolfe step taken, step_size: {alpha_t}")
            # Step 18
            if alpha_t == 1:
                St = w_t
            else:
                St[w_t_idx] = 1
        # Step 19 - end if
        active_set_size_list.append(np.sum(np.abs(St) >= 0.0001))


        # Step 20
        u = u - alpha_t * d_t

    # Step 21 - end for
        # Track time
        it_time = time.time() - t_start
        total_time = total_time + it_time
        CPU_time_list.append(total_time)

    radius = np.sqrt(-dual)
    center = A @ u

    logging.info("\n-----------Away step Frank Wolfe finished!--------------")
    logging.info(f"center: {center} and radius: {radius} ")
    logging.info(f"Last value of dual function {dual:.3e}")
    logging.info(f"Last value of dual gap  {dual_gap:.3e}")
    logging.info(f"Total CPU time {total_time:.3e}")
    logging.info(f"Number of non-zero components of x = {active_set_size_list[-1]}")
    logging.info(f"Number of iterations = {len(dual_list)}")
    return center,radius, active_set_size_list ,dual_gap_list, dual_list, CPU_time_list

def find_max_dist_idx(A_mat, point):

    # Calculate Euclidean distances between the first point and all other points
    euclidean_distances = np.linalg.norm(A_mat - point[:, np.newaxis], axis=0)

    # Find the maximum Euclidean distance point's index
    return np.argmax(euclidean_distances)

def calculate_delta(cntr, furthest_point, gamma):
    # Calculate termination criterion delta which should be greater than (1 + eps) - 1
    gamma += 1e-10  # To avoid division by 0
    euclidean_distances = np.linalg.norm(furthest_point - cntr) / gamma - 1
    return np.max(euclidean_distances)

def one_plus_eps_MEB_approximation(A, eps, max_iter=1000):
    logging.info("(1+epsilon)-approximation algorithm first iteration started!")

    #initialize output lists
    active_set_size_list = []
    dual_gap_list = [0]
    dual_list = [0]
    total_time = 0
    CPU_time_list = [0]

    # shape A: n*m, n is the dimension of the points, m is the number of points
    n_A, m_A = np.shape(A)
    logging.info(f"Dataset size: {m_A} points, each {n_A}-dimensional.")
    # alpha_k = 0 # Initialize step size
    # Step 1
    a = find_max_dist_idx(A, A[:, 0])  # get the point index furthest from first point in A (index 0)
    b = find_max_dist_idx(A, A[:, a])  # get the point index furthest from a in A
    # Step 2
    u = np.zeros(m_A)
    # Step 3
    u[a] = 0.5
    u[b] = 0.5
    # Step 4 - Create active set , here active set includes the indices, in FW version it includes weights of indices
    Xk = [a, b]
    active_set_size_list.append(len(Xk))
    # Step 5 - Initialize center
    c = A @ u  # c should be n dimensional like points a
    # Step 6 - objective function
    dual = dual_function(A, u)
    logging.info(f"Dual function value found: {dual} ")
    dual_gap = dual_list[-1] - 0
    logging.info(f"Dual gap value found: {dual_gap} ")
    r2 = -dual  # r^2 is gamma -- radius^2
    # Step 7
    K = find_max_dist_idx(A, c)  # get the point index furthest from center c
    # Step 8  - Delta is termination criterion delta > (1+eps)^2 -1
    delta = calculate_delta(c, A[:, K], r2)
    # Step 9 - Initialize iterations
    k = 0
    # Step 10
    while True:
        t_start = time.time()
        # Step 11 - loop

        logging.info(f"--------------Iteration {k} -----------------")
        if delta <= (1+eps)**2 - 1:
            logging.info(f"Stopping condition delta <= (1+eps)**2 - 1 is met!")
            it_time = time.time() - t_start
            total_time = total_time + it_time
            CPU_time_list.append(total_time)
            active_set_size_list.append(len(Xk))
            dual = dual_function(A, u)
            logging.info(f"Dual function value found: {dual} ")
            dual_list.append(dual)
            dual_gap = dual_list[-1] - dual_list[-2]
            logging.info(f"Dual gap value found: {dual_gap} ")
            dual_gap_list.append(dual_gap)
            break

        elif k > max_iter:
            logging.info(f"Stopping condition max iterations is met!")
            it_time = time.time() - t_start
            total_time = total_time + it_time
            CPU_time_list.append(total_time)
            active_set_size_list.append(len(Xk))
            dual = dual_function(A, u)
            logging.info(f"Dual function value found: {dual} ")
            dual_list.append(dual)
            dual_gap = dual_list[-1] - dual_list[-2]
            logging.info(f"Dual gap value found: {dual_gap} ")
            dual_gap_list.append(dual_gap)
            break

        # Step 12
        alpha_k = delta/(2 * (1 + delta))
        # Step 13
        k = k + 1
        # Step 14 - Update u, use convex combination of u and unit simplex of index K
        eK = np.zeros(m_A)
        eK[K] = 1
        u = (1-alpha_k)*u + alpha_k * eK
        # Step 15 - Update center, use convex combination of previous center and furthest point aK
        c = (1-alpha_k)*c + alpha_k * A[:, K]
        # Step 16 - Update active set
        Xk.append(K)
        active_set_size_list.append(len(Xk))
        # Step 17  - Update gamma
        dual = dual_function(A, u)
        logging.info(f"Dual function value found: {dual} ")
        dual_list.append(dual)
        dual_gap = dual_list[-1] - dual_list[-2]
        logging.info(f"Dual gap value found: {dual_gap} ")
        dual_gap_list.append(dual_gap)
        r2 = -dual
        # Step 18  - Update K (index of the furthest point in A from c)
        K = find_max_dist_idx(A, c)
        # Step 19
        delta = calculate_delta(c, A[:, K], r2)
        # Step 20 - end loop
        # Track time
        it_time = time.time() - t_start
        total_time = total_time + it_time
        CPU_time_list.append(total_time)
    # Step 21 - Output
    approx_radius = np.sqrt((1 + delta) * r2)



    logging.info("\n-----------(1+epsilon)-approximation FW algorithm finished!--------------")
    logging.info(f"center: {c} and radius: {approx_radius} ")
    logging.info(f"Last value of dual function {dual:.3e}")
    logging.info(f"Last value of dual gap  {dual_gap:.3e}")
    logging.info(f"Total CPU time {total_time:.3e}")
    logging.info(f"Number of non-zero components of x = {active_set_size_list[-1]}")
    logging.info(f"Number of iterations = {k}")

    return c,approx_radius, active_set_size_list ,dual_gap_list, dual_list, CPU_time_list



