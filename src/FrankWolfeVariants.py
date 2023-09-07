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

def LMO(grad):
    return np.argmin(grad)

def compute_vertex_away_or_FW_local(active_set_weights, gradient, dim,
                                    localFW=False):
    active_idxs_list = np.where(active_set_weights > 0)[0]
    if localFW:
        min_grad_idx = np.argmin(gradient[active_idxs_list])  # Find min_grad_index from active set (St)
        v_t_idx = active_idxs_list[min_grad_idx]  # Set local FW index from active set (St)
    else:
        max_grad_idx = np.argmax(gradient[active_idxs_list])  # Find max_grad_index from active set (St)
        v_t_idx = active_idxs_list[max_grad_idx]  # Set away vertex index from active set (St)
    v_t = np.zeros(dim)  # Create m-dimensional array and set v_t = 1 and others to 0
    v_t[v_t_idx] = 1.0
    return v_t, v_t_idx

def awayStep_FW(A, eps, line_search_strategy,
                max_iter=1000):
    """

    Algorithm 1 by Lacoste-Julien

    Args:
        A: nxm matrix, the columns represent points, and the rows represent features
        eps: FW gap stopping criterion
        max_iter: max number of iterations
        line_search_strategy: a string specifying the strategy to be used for line search

    Returns: a dictionary containing:
        center - coordinates of the center of the MEB
        radius - length of the radius of the MEB
        number_iterations - number of iterations
        active_set_size_list - list containing the size of the active set per iteration
        dual_gap_list - list containing the FW gap per iteration
        dual_list - list containing the objective function value per iteration
        CPU_time_list - list containing the CPU time per iteration (as a cumulative list)

    """
    logging.info("Away Step Frank Wolfe algorithm first iteration started!")

    # Initialize output lists
    active_set_size_list = []
    dual_gap_list = []
    dual_list = []
    total_time = 0
    CPU_time_list = []

    dual_t = 0
    dual_gap_t = 0

    number_AW = 0
    number_drop = 0
    number_FW = 0

    n, m = A.shape
    logging.info(f"Dataset size: {m} points, each {n}-dimensional.")
    # Initial solution
    u_t = np.zeros(m)
    u_t[0] = 1e0

    # Step 1 - Initialization
    S_t = np.zeros(m)
    S_t[np.where(u_t > 0)[0]] = 1  # Initialize active set St

    # Step 2 - Loop
    for iteration in range(max_iter):
        t_start = time.time()
        logging.info(f"\n--------------Iteration {iteration} -----------------")

        # Objective function
        dual_t = compute_dual_function(A, u_t)
        logging.info(f"Dual function value found: {dual_t} ")
        dual_list.append(dual_t)

        # Gradient evaluation
        grad = compute_gradient(A, u_t)

        # Step 3 - Solution of FW problem
        s_t_idx = LMO(grad)  # Find the index which makes gradient min
        s_t = np.zeros(m)  # Create m-dimensional array and set e_i = 1
        s_t[s_t_idx] = 1.0
        logging.info(f"FW min grad unit simplex index found: {s_t_idx}")

        # Calculate FW direction
        d_FW = s_t - u_t

        # Step 4 (Lacoste-Julien algo1) - Solution of Away Step
        v_t, v_t_idx = compute_vertex_away_or_FW_local(S_t, grad, m)
        logging.info(f"Away step max grad unit simplex index found: {v_t_idx}")

        # Calculate Away Step direction
        d_AW = u_t - v_t

        # Step 5 - If FW gap is small enough, break
        g_FW = -grad.T @ d_FW  # FW gap
        dual_gap_list.append(g_FW)
        if g_FW  <= eps:
            logging.info(f"Stopping condition gap < epsilon is met!")
            iter_time = time.time() - t_start
            total_time = total_time + iter_time
            CPU_time_list.append(total_time)
            active_set_size_list.append(int(np.sum(np.abs(S_t) >= 0.0001)))
            break

        # Step 6 - Compare FW gap and AW gap
        g_AW = -grad.T @ d_AW  # Calculate Away step gap
        if g_FW >= g_AW:
            # Step 7 - Set direction at iteration k as FW direction
            d_t = d_FW
            alpha_max = 1  # Set max step-size
            logging.info(f"Choose Frank Wolfe direction, step_size_max: {alpha_max}")
            fw_check = True
        # Step 8 - Else
        else:
            # Step 9 - Choose away direction and max feasible step-size
            d_t = d_AW
            alpha_max = S_t[v_t_idx] / (1 - S_t[v_t_idx])  # Use the value of unit simplex highest grad coordinate
            logging.info(f"Choose Away Step  direction, step_size_max : {alpha_max}")
            fw_check = False
        # Step 10 - End if

        # Step 11 - Calculate step size using line search
        if line_search_strategy == 'golden_search':
            alpha_t = golden_section_search(compute_dual_function, A, u_t, d_t, a=0, b=alpha_max)
        else:
            alpha_t = 2 / (iteration + 2)
        alpha_t = max(0.0, min(alpha_t, alpha_max))
        logging.info(f"Step size is set. --> alpha_t: {alpha_t}")

        # Step 12
        u_t = u_t + alpha_t * d_t

        # Step 13 - Update active set (Lacoste-Julien, page 4, 1st paragraph)
        if fw_check:  # FW step used
            number_FW += 1
            S_t = (1 - alpha_t) * S_t
            S_t[s_t_idx] = S_t[s_t_idx] + alpha_t
            # Exceptional case: Step-size of 1, this collapses the active set!
            if alpha_t > 1 - np.spacing(1):  # Take a full step
                S_t = np.zeros(m)
                S_t[s_t_idx] = 1
        else:  # Away step used
            S_t = (1 + alpha_t) * S_t
            number_AW += 1
            if abs(alpha_t - alpha_max) < 10 * np.spacing(1):  # Drop step
                number_drop += 1
                S_t[v_t_idx] = 0
            else:
                S_t[v_t_idx] = S_t[v_t_idx] - alpha_t
        active_set_size_list.append(int(np.sum(np.abs(S_t) >= 0.0001)))

        # Step 14 end for

        iter_time = time.time() - t_start
        total_time = total_time + iter_time
        CPU_time_list.append(total_time)

    radius = np.sqrt(-dual_t)
    center = A @ u_t

    logging.info("\n-----------Away step Frank Wolfe finished!--------------")
    logging.info(f"Center: {center} and Radius: {radius}")
    logging.info(f"Last value of dual function: {dual_t:.3e}")
    logging.info(f"Last value of FW gap: {dual_gap_t:.3e}")
    logging.info(f"Total CPU time: {total_time:.3e}")
    logging.info(f"Active set size: {active_set_size_list[-1]}")
    logging.info(f"Number of iterations: {len(dual_list)}")
    logging.info(f"FW steps: {number_FW}")
    logging.info(f"Away steps: {number_AW}")
    logging.info(f"Drop steps: {number_drop}")

    output = {"center": center,
              "radius": radius,
              "number_iterations": len(dual_list),
              "active_set_size_list": active_set_size_list,
              "dual_gap_list": dual_gap_list,
              "dual_list": dual_list,
              "CPU_time_list": CPU_time_list}
    return output

def blendedPairwise_FW(A, eps, line_search_strategy,
                       max_iter=1000):
    """

    Algorithm 1 by Tsuji

    Args:
        A: nxm matrix, the columns represent points, and the rows represent features
        eps: FW gap stopping criterion
        max_iter: max number of iterations
        line_search_strategy: a string specifying the strategy to be used for line search

    Returns: a dictionary containing:
        center - coordinates of the center of the MEB
        radius - length of the radius of the MEB
        number_iterations - number of iterations
        active_set_size_list - list containing the size of the active set per iteration
        dual_gap_list - list containing the FW gap per iteration
        dual_list - list containing the objective function value per iteration
        CPU_time_list - list containing the CPU time per iteration (as a cumulative list)

    """
    logging.info("Blended Pairwise Frank Wolfe algorithm first iteration started!")

    # Initialize output lists
    active_set_size_list = []
    dual_gap_list = []
    dual_list = []
    total_time = 0
    CPU_time_list = []

    n, m = A.shape
    logging.info(f"Dataset size: {m} points, each {n}-dimensional.")

    dual_t = 0
    FW_gap_t = eps + 1.0
    number_FW = 0
    number_descent = 0
    number_drop = 0

    # Initial solution
    u_t = np.zeros(m)
    u_t[0] = 1e0

    # Step 1
    S_t = np.zeros(m)
    S_t[np.where(u_t > 0)[0]] = 1  # Initialize the active set St

    # Step 2 - Loop
    iteration = 0
    while iteration < max_iter and FW_gap_t > eps:
        iteration += 1
        t_start = time.time()
        logging.info(f"\n--------------Iteration {iteration} -----------------")

        # Objective function
        dual_t = compute_dual_function(A, u_t)
        logging.info(f"Dual function value found: {dual_t} ")
        dual_list.append(dual_t)

        # Gradient evaluation
        grad_t = compute_gradient(A, u_t)

        # Step 3 - Away vertex
        a_t, a_t_idx = compute_vertex_away_or_FW_local(S_t, grad_t, m)
        logging.info(f"Away step max grad unit simplex index found: {a_t_idx}")

        # Step 4 - Local FW
        s_t, s_t_idx = compute_vertex_away_or_FW_local(S_t, grad_t, m, localFW=True)
        logging.info(f"Local FW min grad unit simplex index found: {s_t_idx}")

        # Step 5 - Global FW
        w_t_idx = np.argmin(grad_t)  # Find the index which makes gradient min - search in all vertices
        w_t = np.zeros(m)  # Create m-dimensional array and set w_t = 1 and others to 0
        w_t[w_t_idx] = 1.0
        logging.info(f"Global FW min grad unit simplex index found: {w_t_idx}")

        FW_gap_t = max(0.0, grad_t.T @ (u_t - w_t))
        logging.info(f"Dual gap value found: {FW_gap_t} ")
        dual_gap_list.append(FW_gap_t)

        # Step 6 - Compare local pairwise gap and FW gap
        local_pairwise_gap = max(0.0, grad_t.T @ (a_t - s_t))
        if local_pairwise_gap >= FW_gap_t:
            # Step 7
            d_t = a_t - s_t  # Away vertex - local FW
            # Step 8 - Max step size
            alpha_max = u_t[a_t_idx]
            # Step 9 - Calculate step size using line search
            if line_search_strategy == 'golden_search':
                alpha_t = golden_section_search(compute_dual_function, A, u_t, d_t, a=0, b=alpha_max)
            else:
                alpha_t = 2 / (iteration + 2)
            alpha_t = max(0.0, min(alpha_t, alpha_max))
            # Step 10
            if alpha_t < alpha_max:
                # Step 11
                S_t = S_t  # Descent step, do not update St
                number_descent += 1
                logging.info(f"Descent step taken, step_size: {alpha_t}")
            # Step 12
            else:
                # Step 13
                S_t[a_t_idx] = 0  # Drop step (The away vertex is removed from the active set St => set weight = 0)
                # Reached maximum of lambda -> dropping away vertex
                number_drop += 1
                logging.info(f"Drop step taken, step_size: {alpha_t}")
        # Step 14 - End if
        # Step 15 - Else
        else:  # The local pairwise gap is smaller than the FW gap => take a FW step
            # Step 16
            d_t = u_t - w_t  # u weights - global FW
            number_FW += 1
            # Step 17
            alpha_max = 1.0
            if line_search_strategy == 'golden_search':
                alpha_t = golden_section_search(compute_dual_function, A, u_t, d_t, a=0, b=alpha_max)
            else:
                alpha_t = 2 / (iteration + 2)
            alpha_t = max(0.0, min(alpha_t, alpha_max))
            logging.info(f"Choose Frank Wolfe step taken, step_size: {alpha_t}")
            # Step 18
            if alpha_t == 1:
                S_t = w_t
            else:
                S_t[w_t_idx] = 1
        # Step 19 - End if
        active_set_size_list.append(int(np.sum(np.abs(S_t) >= 0.0001)))

        # Step 20
        u_t = u_t - alpha_t * d_t

        # Track time
        iter_time = time.time() - t_start
        total_time = total_time + iter_time
        CPU_time_list.append(total_time)

        # Step 21 - End for

    radius = np.sqrt(-dual_t)
    center = A @ u_t

    logging.info("\n-----------Blended Pairwise Frank Wolfe finished!--------------")
    logging.info(f"Center: {center} and Radius: {radius}")
    logging.info(f"Last value of dual function {dual_t:.3e}")
    logging.info(f"Last value of dual gap {FW_gap_t:.3e}")
    logging.info(f"Total CPU time {total_time:.3e}")
    logging.info(f"Active set size = {active_set_size_list[-1]}")
    logging.info(f"Number of iterations = {len(dual_list)}")
    logging.info(f"FW steps: {number_FW}")
    logging.info(f"Descent steps: {number_descent}")
    logging.info(f"Drop steps: {number_drop}")

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

def one_plus_eps_MEB_approximation(A, eps,
                                   max_iter=1000):
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
    active_set_size_list.append(int(len(Xk)))
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
        active_set_size_list.append(int(len(Xk)))
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
