# -*- coding: utf-8 -*-
"""
Cat Water Phobia Optimizer (CWPO)
Inspired by the survival strategies of stray cats
"""

import numpy as np
import time
from solution import solution

def levy_flight(lam):
    """Generate a Levy flight step."""
    sigma = (np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2) /
             (np.math.gamma((1 + lam) / 2) * lam * 2 ** ((lam - 1) / 2))) ** (1 / lam)
    u = np.random.normal(0, sigma, 1)
    v = np.random.normal(0, 1, 1)
    step = u / abs(v) ** (1 / lam)
    return step[0]

def hazard_function(alpha, beta, cat_pos, local_minima, levy_lambda):
    """
    Calculate the environmental hazard.
    EH = alpha * abs(C_i - LM) + beta * Levy(lambda)
    """
    distance_to_minima = np.abs(cat_pos - local_minima)
    levy_randomness = beta * levy_flight(levy_lambda)
    return alpha * distance_to_minima + levy_randomness

def add_escape_strategy(Positions, fitness_values, global_best_pos, escape_factor=0.1):
    """
    Introduce an escape mechanism to prevent getting trapped in local minima.
    """
    perturbation = escape_factor * np.random.uniform(-1, 1, Positions.shape)
    Positions += perturbation * (Positions - global_best_pos)
    return Positions

def CWPO(objf, lb, ub, dim, SearchAgents_no, Max_iter, alpha_start=1.5, alpha_end=0.5,
                  beta_start=0.5, beta_end=0.1, levy_lambda=1.5, elite_fraction=0.1, escape_factor=0.1):
    """
    Enhanced CWPO with dynamic parameters, elite preservation, adaptive mechanisms, and escape strategy.
    objf: Objective function
    lb: Lower bound
    ub: Upper bound
    dim: Number of dimensions
    SearchAgents_no: Population size
    Max_iter: Maximum iterations
    alpha_start: Initial hazard impact scaling factor
    alpha_end: Final hazard impact scaling factor
    beta_start: Initial hazard fluctuation factor
    beta_end: Final hazard fluctuation factor
    levy_lambda: Levy flight parameter
    elite_fraction: Fraction of elite solutions to preserve
    escape_factor: Factor to control the escape strategy
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

        # Update local minima dynamically
        local_minima = Alpha_pos.copy()

        for i in range(SearchAgents_no):
            # Calculate environmental hazard
            hazard_effect = hazard_function(
                alpha=alpha,
                beta=beta,
                cat_pos=Positions[i],
                local_minima=local_minima,
                levy_lambda=levy_lambda
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

        # Escape strategy to avoid local minima
        Positions = add_escape_strategy(Positions, Convergence_curve, Alpha_pos, escape_factor)

        Convergence_curve[l] = Alpha_score

        if (l + 1) % 500 == 0:
            print(f"At iteration {l + 1}, the best fitness is {Alpha_score}")

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "Enhanced CWPO with Escape Strategy"
    s.bestIndividual = Alpha_pos
    s.objfname = objf.__name__

    return s
