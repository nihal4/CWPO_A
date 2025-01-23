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

def CWPO(objf, lb, ub, dim, SearchAgents_no, Max_iter, alpha_start=1.5, alpha_end=0.5,
                  beta_start=0.5, beta_end=0.1, sigma_start=1.0, sigma_end=0.1, elite_fraction=0.1):
    """
    Enhanced CWPO with dynamic parameters, elite preservation, and adaptive mechanisms.
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

    # Elite archive
    elite_count = int(SearchAgents_no * elite_fraction)
    Elite_archive = np.zeros((elite_count, dim))
    Elite_scores = np.full(elite_count, float("inf"))

    print('CWPO is optimizing "' + objf.__name__ + '"')
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(Max_iter):
        # Dynamic parameter tuning
        alpha = alpha_start - (alpha_start - alpha_end) * (l / Max_iter)
        beta = beta_start - (beta_start - beta_end) * (l / Max_iter)
        sigma = sigma_start - (sigma_start - sigma_end) * (l / Max_iter)

        # Update local minima dynamically
        local_minima = Alpha_pos.copy()

        for i in range(SearchAgents_no):
            # Calculate environmental hazard
            hazard_effect = hazard_function(alpha, beta, Positions[i], local_minima, sigma)

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

            # Update elite archive
            if fitness < max(Elite_scores):
                max_idx = np.argmax(Elite_scores)
                Elite_archive[max_idx] = Positions[i].copy()
                Elite_scores[max_idx] = fitness

        # Elite preservation
        for j in range(elite_count):
            rand_idx = np.random.randint(SearchAgents_no)
            if Elite_scores[j] < objf(Positions[rand_idx]):
                Positions[rand_idx] = Elite_archive[j]

        Convergence_curve[l] = Alpha_score

        if (l + 1) % 500 == 0:
            print(f"At iteration {l + 1}, the best fitness is {Alpha_score}")

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "Enhanced CWPO"
    s.bestIndividual = Alpha_pos
    s.objfname = objf.__name__

    return s

