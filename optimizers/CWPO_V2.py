# -*- coding: utf-8 -*-
"""
Cat Water Phobia Optimizer (CWPO)
Inspired by the survival strategies of stray cats
"""

import numpy as np
import time
from solution import solution

def gaussian_random_walk(sigma):
    """Generate a Gaussian random step."""
    return np.random.normal(0, sigma, 1)[0]

def hazard_function(alpha, beta, cat_pos, local_minima, sigma):
    """
    Calculate the environmental hazard.
    EH = alpha * abs(C_i - LM) + beta * Gaussian(sigma)
    """
    distance_to_minima = np.abs(cat_pos - local_minima)
    gaussian_randomness = beta * gaussian_random_walk(sigma)
    return alpha * distance_to_minima + gaussian_randomness

def CWPO(objf, lb, ub, dim, SearchAgents_no, Max_iter, alpha=1.5, beta=0.5, sigma=1.0):
    """
    Cat Water Phobia Optimizer (CWPO)
    objf: Objective function
    lb: Lower bound
    ub: Upper bound
    dim: Number of dimensions
    SearchAgents_no: Population size
    Max_iter: Maximum iterations
    alpha: Hazard impact scaling factor
    beta: Hazard fluctuation factor
    sigma: Standard deviation for Gaussian random walk
    """

    # Initialize population
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = np.random.uniform(0, 1, SearchAgents_no) * (ub - lb) + lb

    # Initialize the best solution
    Alpha_pos = np.zeros(dim)
    Alpha_score = float("inf")

    Convergence_curve = np.zeros(Max_iter)
    s = solution()

    # Start optimization
    print('CWPO is optimizing "' + objf.__name__ + '"')
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(Max_iter):
        # Dynamically calculate the local minima (best position so far)
        local_minima = Alpha_pos.copy()

        for i in range(SearchAgents_no):
            # Calculate environmental hazard
            hazard_effect = hazard_function(
                alpha=alpha,
                beta=beta,
                cat_pos=Positions[i],
                local_minima=local_minima,
                sigma=sigma
            )

            # Update position using exploration and hazard exploitation
            Positions[i] = Positions[i] - hazard_effect

            # Ensure position stays within bounds
            Positions[i] = np.clip(Positions[i], lb, ub)

            # Calculate fitness
            fitness = objf(Positions[i])

            # Update the best solution
            if fitness < Alpha_score:
                Alpha_score = fitness
                Alpha_pos = Positions[i].copy()

        Convergence_curve[l] = Alpha_score

        if (l + 1) % 500 == 0:
            print(f"At iteration {l + 1}, the best fitness is {Alpha_score}")

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "CWPO"
    s.bestIndividual = Alpha_pos
    s.objfname = objf.__name__

    return s
