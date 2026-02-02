import numpy as np


def read_dataset(dataset, from_real):
    raise NotImplementedError("implement dataset reading")


def simulate_forward_kinematics(young_modulues, motor_angles):
    raise NotImplementedError("implement dataset reading")


def calibrate_young(dataset, from_real=False):

    # read dataset of motor angle - end effector position pairs
    dataset_pairs = read_dataset(dataset, from_real)

    delta = 1e-5  # finite-diff parameter
    alpha = 1e-2  # stepsize

    E = 1.0  # starting value of E
    converged = False
    msg = "reached maximum number of iterations"

    tol = 1e-8
    max_iter = 1000
    for i in range(max_iter):

        for m, p in dataset_pairs:
            # run forward simulator with E, m
            p_sim = simulate_forward_kinematics(E, m)
            f_sim = np.linalg.norm(p_sim - p)

            p_sim_delta = simulate_forward_kinematics(E + delta, m)
            f_sim_delta = np.linalg.norm(p_sim_delta - p)

            # calculate gradient using forward difference
            gradient = (f_sim_delta - f_sim) / delta

            # update Young modulus
            E += E + alpha * gradient

            # check convergence
            if np.abs(gradient) <= tol:
                msg = "converged in gradient norm"

    results = {"msg": msg, "success": converged}
    return E, results
