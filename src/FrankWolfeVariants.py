import numpy as np
import math
import time
from src.logger import logging

def compute_dual_function(A, u):
    first_term = u.T @ A.T @ A @ u
    Z = np.sum(A**2, axis=0)
    second_term = Z.T @ u
    return first_term - second_term

def compute_dual_Yildirim(A, u_vector):
    n, m = A.shape
    first_term_sum = 0  # Refers to the first term of the equation
    second_term_product = np.zeros(n)  # Refers to the vector in the second term
    for i in range(m):
        col = A[:, i]
        first_term_sum += u_vector[i] * np.dot(col.T, col)  # Transpose is not needed for vectors multiplication
        second_term_product += u_vector[i] * col
    return first_term_sum - np.dot(second_term_product.T, second_term_product)

def compute_gradient(A, u):
    Z = np.sum(A ** 2, axis=0)
    return 2 * A.T @ A @ u - Z

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

    # Initialize output lists
    active_set_size_list = []
    dual_gap_list = []
    dual_list = []
    total_time = 0
    CPU_time_list = []

    dual_k = 0
    dual_gap = 0

    n, m = A.shape
    logging.info(f"Dataset size: {m} points, each {n}-dimensional.")
    # initial solution
    u_k = np.zeros(m)
    u_k[0] = 1e0

    # Step 1 - Initialization
    St = np.zeros(m)
    St[np.where(u_k > 0)[0]] = 1  # active set St -- initialize

    # Step 2 - Loop
    for iteration in range(max_iter):
        t_start = time.time()
        logging.info(f"\n--------------Iteration {iteration} -----------------")

        # Objective function
        dual_k = compute_dual_function(A, u_k)
        logging.info(f"Dual function value found: {dual_k} ")
        dual_list.append(dual_k)

        # Gradient evaluation
        grad = compute_gradient(A, u_k)

        # Step 3 - Solution of FW problem
        s_t_idx = np.argmin(grad)  # Find the index which makes gradient min
        s_t = np.zeros(m)  # Create m-dimensional array and set e_i = 1
        s_t[s_t_idx] = 1.0
        logging.info(f"FW min grad unit simplex index found: {s_t_idx}")

        # Calculate FW direction
        d_FW = s_t - u_k

        # Step 4 (Lacoste-Julien algo1) - Solution of Away Step
        active_idxs_list = np.where(St > 0)[0]
        max_grad_idx = np.argmax(grad[active_idxs_list])  # Find max_grad_index from active set St
        v_t_idx = active_idxs_list[max_grad_idx]  # Set away vertex index from active set St
        v_t = np.zeros(m)  # Create m-dimensional array and set v_t = 1 and others to 0
        v_t[v_t_idx] = 1.0
        logging.info(f"Away step max grad unit simplex index found: {v_t_idx}")

        # Calculate Away Step direction
        d_AW = u_k - v_t

        # Step 5 - If FW gap is small enough, break
        g_FW = -grad.T @ d_FW
        dual_gap_list.append(g_FW)
        if g_FW  <= eps:
            logging.info(f"Stopping condition gap < epsilon is met!")
            it_time = time.time() - t_start
            total_time = total_time + it_time
            CPU_time_list.append(total_time)
            active_set_size_list.append(np.sum(np.abs(St) >= 0.0001))
            break

        # Step 6 - Compare FW gap and AW gap
        g_AW = -grad.T @ d_AW  # Calculate Away step gap
        if g_FW >= g_AW:
            # Step 7 - Set direction at iteration k as FW direction
            d_t = d_FW
            alpha_max = 1  # Set max step_size
            logging.info(f"Choose Frank Wolfe direction, step_size_max: {alpha_max}")
            fw_check = True

        # Step 8 - Else
        else:
            # Step 9 - Choose away direction and max feasible step-size
            d_t = d_AW
            alpha_max = St[v_t_idx] / (1 - St[v_t_idx])  # Use the value of unit simplex highest grad coordinate
            logging.info(f"Choose Away Step  direction, step_size_max : {alpha_max}")
            fw_check = False

        # Step 10 - End if

        # Step 11 - Calculate step size using line search
        if line_search_strategy == 'line_search_strategy':
            alpha_t = golden_section_search(compute_dual_function, A, u_k, d_t, a=0, b=alpha_max)
        else:
            alpha_t = 2 / (iteration + 2)
        alpha_t = max(0.0, min(alpha_t, alpha_max))
        logging.info(f"Step size is set. --> alpha_t: {alpha_t}")

        # Step 12
        u_k = u_k + alpha_t * d_t

        # Step 13 - Update active set (Lacoste-Juline pg.4 1st paragraph)
        if fw_check:  # fw step used
            St = (1 - alpha_t) * St
            St[s_t_idx] = St[s_t_idx] + alpha_t
        else:  # away step used
            St = (1 + alpha_t) * St
            St[v_t_idx] = St[v_t_idx] - alpha_t
        active_set_size_list.append(np.sum(np.abs(St) >= 0.0001))
        # Step 14 end for

        it_time = time.time() - t_start
        total_time = total_time + it_time
        CPU_time_list.append(total_time)

    radius = np.sqrt(-dual_k)
    center = A @ u_k
    # (256 x 2) x (2

    logging.info("\n-----------Away step Frank Wolfe finished!--------------")
    logging.info(f"center: {center} and radius: {radius} ")
    logging.info(f"Last value of dual function {dual_k:.3e}")
    logging.info(f"Last value of dual gap  {dual_gap:.3e}")
    logging.info(f"Total CPU time {total_time:.3e}")
    logging.info(f"Number of non-zero components of x = {active_set_size_list[-1]}")
    logging.info(f"Number of iterations = {len(dual_list)}")

    output = {"center": center,
              "radius": radius,
              "number_iterations": len(dual_list),
              "active_set_size_list": active_set_size_list,
              "dual_gap_list": dual_gap_list,
              "dual_list": dual_list,
              "CPU_time_list": CPU_time_list}
    return output

def blendedPairwise_FW(A, eps, max_iter=1000):  # Tsuji Algorithm 1
    logging.info("Blended Pairwise Frank Wolfe algorithm first iteration started!")

    # Initialize output lists
    active_set_size_list = []
    dual_gap_list = []
    dual_list = []
    total_time = 0
    CPU_time_list = []

    n, m = A.shape
    # alpha_max = 1  # max step_size
    # alpha_t = 0  # step_size at iteration t
    logging.info(f"Dataset size: {m} points, each {n}-dimensional.")

    dual = 0
    dual_gap = 0
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
        dual = compute_dual_function(A, u)
        logging.info(f"Dual function value found: {dual} ")
        dual_list.append(dual)
        if iteration > 0:
            dual_gap = dual_list[-1] - dual_list[-2]
            logging.info(f"Dual gap value found: {dual_gap} ")
            dual_gap_list.append(dual_gap)

        # gradient evaluation
        grad = compute_gradient(A, u)

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
            alpha_t  = golden_section_search(compute_dual_function, A, u, d_t, a=0, b=alpha_max)
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
            alpha_t  = golden_section_search(compute_dual_function, A, u, d_t, a=0, b=1)
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

    # this is a trick to make dual_gap list size equal to other lists
    dual_gap_list.append(dual_gap)

    radius = np.sqrt(-dual)
    center = A @ u

    logging.info("\n-----------Away step Frank Wolfe finished!--------------")
    logging.info(f"center: {center} and radius: {radius} ")
    logging.info(f"Last value of dual function {dual:.3e}")
    logging.info(f"Last value of dual gap  {dual_gap:.3e}")
    logging.info(f"Total CPU time {total_time:.3e}")
    logging.info(f"Number of non-zero components of x = {active_set_size_list[-1]}")
    logging.info(f"Number of iterations = {len(dual_list)}")

    output = {"center": center,
              "radius": radius,
              "number_iterations": len(dual_list),
              "active_set_size_list": active_set_size_list,
              "dual_gap_list": dual_gap_list,
              "dual_list": dual_list,
              "CPU_time_list": CPU_time_list}
    return output

def find_furthest_point_idx(A_mat, point):
    # Calculate squared Euclidean distances between the first point and all other points
    squared_distances = np.sum((A_mat - point[:, np.newaxis]) ** 2, axis=0)
    return np.argmax(squared_distances)

def calculate_delta(cntr, furthest_point, gamma):
    # Calculate termination criterion delta which should be greater than (1 + eps) - 1
    if gamma == 0:
        gamma = 1e-10  # To avoid division by 0
    # euclidean_distance = np.linalg.norm(furthest_point - cntr)
    # delta = euclidean_distance ** 2 / gamma - 1
    squared_distance = np.sum((furthest_point - cntr) ** 2)
    delta = squared_distance / gamma - 1
    return delta

def one_plus_eps_MEB_approximation(A, eps, max_iter=1000):
    logging.info("(1+epsilon)-approximation algorithm first iteration started!")

    # Initialize output lists
    active_set_size_list = []
    dual_list = []
    delta_list = []
    total_time = 0
    CPU_time_list = [0]

    n_A, m_A = np.shape(A)
    logging.info(f"Dataset size: {m_A} points, each {n_A}-dimensional.")

    # Step 1
    alpha = find_furthest_point_idx(A, A[:, 0])  # Get the index of the point furthest from first point in A (index 0)
    beta = find_furthest_point_idx(A, A[:, alpha])  # Get the index of the point furthest from point a in A
    # Step 2
    u_k = np.zeros(m_A)
    # Step 3
    u_k[alpha] = 0.5
    u_k[beta] = 0.5
    # Step 4 - Create active set (Here active set includes the indices, in FW version it includes weights of indices)
    Xk = [alpha, beta]
    active_set_size_list.append(len(Xk))
    # Step 5 - Initialize center
    c_k = A @ u_k  # c should be n dimensional like points a
    # Step 6 - objective function
    dual_k = compute_dual_Yildirim(A, u_k)
    logging.info(f"Dual function value found: {dual_k} ")
    dual_list.append(dual_k)
    r2 = dual_k  # r^2 is gamma (radius^2)
    # Step 7
    K = find_furthest_point_idx(A, c_k)  # Get the index of the point furthest from the center c
    # Step 8 - Delta is termination criterion delta > (1+eps)^2 - 1
    delta_k = calculate_delta(c_k, A[:, K], r2)
    logging.info(f"Delta value found: {delta_k} ")
    delta_list.append(delta_k)
    # Step 9 - Initialize iterations
    k = 0
    # Step 10 - Stopping conditions
    deltaHasNotReachedThreshold = delta_k > (1+eps)**2 - 1
    maxIterNotReached = k < max_iter
    # Step 11 - Loop
    while deltaHasNotReachedThreshold and maxIterNotReached:
        t_start = time.time()
        logging.info(f"--------------Iteration {k} -----------------")
        # Step 12
        alpha_k = delta_k/(2 * (1 + delta_k))
        # Step 13
        k = k + 1
        maxIterNotReached = k < max_iter
        # Step 14 - Update u, use convex combination of u and unit simplex of index K
        e_K = np.zeros(m_A)
        e_K[K] = 1
        u_k = (1 - alpha_k) * u_k + alpha_k * e_K
        # Step 15 - Update center, use convex combination of previous center and furthest point aK
        c_k = (1 - alpha_k) * c_k + alpha_k * A[:, K]
        # Step 16 - Update active set
        if K not in Xk:
            Xk.append(K)
        active_set_size_list.append(len(Xk))
        # Step 17 - Update gamma (r2)
        dual_k = compute_dual_Yildirim(A, u_k)
        logging.info(f"Dual function value found: {dual_k} ")
        dual_list.append(dual_k)
        r2 = dual_k
        # Step 18 - Update K (index of the furthest point in A from point c)
        K = find_furthest_point_idx(A, c_k)
        # Step 19
        delta_k = calculate_delta(c_k, A[:, K], r2)
        logging.info(f"Delta value found: {delta_k} ")
        delta_list.append(delta_k)
        deltaHasNotReachedThreshold = delta_k > (1 + eps) ** 2 - 1
        # Track time
        iter_time = time.time() - t_start
        total_time = total_time + iter_time
        CPU_time_list.append(total_time)
        # Step 20 - end loop

    if not deltaHasNotReachedThreshold:
        logging.info(f"Stopping condition delta <= (1+eps)**2 - 1 is met!")
    elif not maxIterNotReached:
        logging.info(f"Stopping condition max iterations is met!")

    approx_radius = np.sqrt((1 + delta_k) * r2)

    logging.info("\n-----------(1+epsilon)-approximation FW algorithm finished!--------------")
    logging.info(f"center: {c_k} and radius: {approx_radius} ")
    logging.info(f"Last value of dual function {dual_k:.3e}")
    logging.info(f"Last value of delta  {delta_k:.3e}")
    logging.info(f"Total CPU time {total_time:.3e}")
    logging.info(f"Number of non-zero components of x = {active_set_size_list[-1]}")
    logging.info(f"Active set: {Xk}")
    logging.info(f"Number of iterations = {k}")

    # Step 21 - Output
    output = {"center": c_k,
              "radius": approx_radius,
              "number_iterations": len(dual_list),
              "active_set_size_list": active_set_size_list,
              "dual_gap_list": delta_list,
              "dual_list": dual_list,
              "CPU_time_list": CPU_time_list}
    return output
