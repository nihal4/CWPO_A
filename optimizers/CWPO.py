# -*- coding: utf-8 -*-
"""
Cat Water Phobia Optimizer (CWPO) with Adaptive Parameter Tuning
"""

import numpy as np
import random
import time
from solution import solution


def levy_flight(lam, scale=1.0):
    """Generate a Levy flight step."""
    sigma = (np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2) /
             (np.math.gamma((1 + lam) / 2) * lam * 2 ** ((lam - 1) / 2))) ** (1 / lam)
    u = np.random.normal(0, sigma, 1)
    v = np.random.normal(0, 1, 1)
    step = scale * (u / abs(v) ** (1 / lam))
    return step[0]


def dynamic_hazard(t, omega, alpha, beta, p, cat_pos, hazard_pos):
    """
    Calculate the environmental hazard at time t.
    EH_i = alpha * ||C_i - O(t)||^p + beta * cos(omega * t)
    """
    distance = np.linalg.norm(cat_pos - hazard_pos) ** p
    periodic_fluctuation = beta * np.cos(omega * t)
    return alpha * distance + periodic_fluctuation


def localized_search(cat_pos, resources, sf):
    """
    Localized search function for nearby resources based on:
    S(C_i, SF) = sum(R_j / ||C_i - R_j||)
    """
    distances = np.linalg.norm(resources - cat_pos, axis=1)
    nearby_indices = distances <= sf
    nearby_resources = resources[nearby_indices]
    nearby_distances = distances[nearby_indices]

    if len(nearby_resources) > 0:
        weighted_sum = np.sum(nearby_resources.T / nearby_distances, axis=1)
        return weighted_sum
    else:
        return np.zeros_like(cat_pos)


def CWPO(objf, lb, ub, dim, SearchAgents_no, Max_iter, alpha=0.8, beta=0.3, omega_freq=0.05, sf= 2.5, levy_lambda=1.5, p=2.0):
    """
    Cat Water Phobia Optimizer (CWPO) with Adaptive Parameter Tuning
    objf: Objective function
    lb: Lower bound
    ub: Upper bound
    dim: Number of dimensions
    SearchAgents_no: Population size
    Max_iter: Maximum iterations
    alpha: Initial hazard impact scaling factor
    beta: Initial hazard fluctuation amplitude
    omega_freq: Frequency of hazard oscillation (f), resulting in omega = 2 * pi * f
    sf: Initial search radius
    levy_lambda: Initial Levy flight parameter
    p: Hazard decay exponent
    """

    # Calculate angular frequency
    omega = 2 * np.pi * omega_freq

    # Initialize population, hazards, and resources
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = np.random.uniform(0, 1, SearchAgents_no) * (ub - lb) + lb

    Hazards = np.random.uniform(lb, ub, (SearchAgents_no, dim))
    Resources = np.random.uniform(lb, ub, (SearchAgents_no, dim))

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
        # Adaptive parameter tuning
        adaptive_alpha = alpha * (1 - l / Max_iter)  # Decrease alpha over time
        adaptive_sf = sf * (1 - l / Max_iter)        # Reduce search radius over time
        adaptive_lambda = levy_lambda * (1 + l / Max_iter)  # Increase Levy randomness over time

        for i in range(SearchAgents_no):
            # Dynamic environmental hazard
            hazard_effect = dynamic_hazard(
                t=l,
                omega=omega,
                alpha=adaptive_alpha,
                beta=beta,
                p=p,
                cat_pos=Positions[i],
                hazard_pos=Hazards[i]
            )

            # Localized search
            local_search = localized_search(Positions[i], Resources, adaptive_sf)

            # Update position using exploration equation
            Positions[i] += (Positions[i] * levy_flight(adaptive_lambda)) - hazard_effect + local_search

            # Ensure position stays within bounds
            Positions[i] = np.clip(Positions[i], lb, ub)

            # Calculate fitness
            fitness = objf(Positions[i])

            # Update the best solution
            if fitness < Alpha_score:
                Alpha_score = fitness
                Alpha_pos = Positions[i].copy()

        Convergence_curve[l] = Alpha_score

        if l % 1 == 0:
            print(f"At iteration {l + 1}, the best fitness is {Alpha_score}")

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "CWPO"
    s.bestIndividual = Alpha_pos
    s.objfname = objf.__name__

    return s
