import numpy as np
import time
from src.logger import logging

# TODO: add logging
# TODO: add maxit
# TODO: keep CPU time
# TODO: check 3 algorithms to be suitable to return necessary lists indicated for graphs.

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

def find_max_dist_idx(A,point):

    # Calculate Euclidean distances between the first point and all other points
    euclidean_distances = np.linalg.norm(A - point[:, np.newaxis], axis=0)

    # Find the maximum Euclidean distance point's index
    return np.argmax(euclidean_distances)

def calculate_delta(center,furthest_point,gamma):
    # calculate termination criterion delta which should be greater than (1+eps)-1

    euclidean_distances = np.linalg.norm(furthest_point-center)/gamma -1
    return np.max(euclidean_distances)



def one_plus_eps_MEB_approximation(A, u, eps, maxit):
    # shape A: n*m , n is dimension points, m number of points
    n, m = np.shape(A)
    alpha_k = 0 # initialize step size
    # step1
    a= find_max_dist_idx(A,A[:,0]) # get the point index furthest from first point in A (index 0)
    b= find_max_dist_idx(A,A[:,a]) # get the point index furthest from a in A

    # step2
    u = np.zeros(m) # zeroize u
    # step3
    u[a] = 0.5
    u[b] = 0.5
    # step4 -- create active set , here active set includes the indices, in FW version it includes weights of indices
    Xk = []
    Xk.append(a)
    Xk.append(b)
    # step5 - initialize center
    c = A @ u # c should be n dimensional like points a
    # step6
    r2 = -dual_function(A,u)  # r^2 is gamma -- radius^2
    # step7
    K = find_max_dist_idx(A,c)  # get the point index furthest from center c
    # step8  -- delta is termination criterion delta > (1+eps)^2 -1
    delta = calculate_delta(c,A[:,K],r2)
    # step9 - initialize iterations
    k = 0
    # step10
    while delta > (1+eps)**2-1:
    # step11 - loop
        # step12
        alpha_k = delta/(2*(1+delta))
        #step 13
        k = k + 1
        #step 14  -- update u, use convex combination of u and unit simplex of index K
        eK = np.zeros(m)
        eK[K] = 1
        u = (1-alpha_k)*u + alpha_k * eK
        #step 15  -- update center, use convex combination of previous center and furthest point aK
        c = (1-alpha_k)*c + alpha_k * A[:,K]
        #step 16  -update active set
        Xk.append(K)
        #step 17  - update gamma
        r2 = -dual_function(A,u)
        #step 18  - update K - index of a point in A which furhest point from c
        K = find_max_dist_idx(A,c)
        #step 19
        delta = calculate_delta(c,A[:,K],r2)
        #step 20 - end loop
    #step 21 - Output
    approx_radius = np.sqrt((1+delta)*r2)
    return c, Xk, u, approx_radius


if __name__ == '__main__':

    # Number of samples
    m = 2 ** 3

    # Dimension of variables
    n = 2 ** 1

    # Max number of iterations
    maxit = 1000

    # initial solution
    u0 = np.zeros(m)
    u0[0] = 1e0

    A = np.random.randn(n, m)

    eps = 1e-6

    c,Xk,u,r = one_plus_eps_MEB_approximation(A,u0,eps,maxit)
    print(f"center : {c}")
    print(f"radius: {r}")
    print(f"active set {sorted(Xk)}")
    print(f"u: {u}")