
# -*- coding: utf-8 -*-
"""
Cat Water Phobia Optimizer (CWPO)
Inspired by the survival strategies of stray cats
"""

import numpy as np
import random
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


def dynamic_hazard(t, omega, alpha, beta, p, cat_pos, hazard_pos):
    """
    Calculate the environmental hazard at time t.
    EH_i = alpha * ||C_i - O(t)||^p + beta * cos(omega * t)
    """
    distance = np.linalg.norm(cat_pos - hazard_pos) ** p
    periodic_fluctuation = beta * np.cos(omega * t)
    return alpha * distance + periodic_fluctuation


def cwpo(objf, lb, ub, dim, SearchAgents_no, Max_iter, alpha=1.0, beta=0.5, omega_freq=0.1, sf=5.0, levy_lambda=1.5, p=-1):
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
    omega_freq: Frequency of hazard oscillation
    sf: Search radius
    levy_lambda: Levy flight parameter
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
        for i in range(SearchAgents_no):
            # Dynamic environmental hazard
            hazard_effect = dynamic_hazard(
                t=l,
                omega=omega,
                alpha=alpha,
                beta=beta,
                p=p,
                cat_pos=Positions[i],
                hazard_pos=Hazards[i]
            )

            # Localized search
            distances_to_resources = np.linalg.norm(Resources - Positions[i], axis=1)
            nearby_resources = Resources[distances_to_resources <= sf]
            if nearby_resources.size > 0:
                local_search = nearby_resources.mean(axis=0)
            else:
                local_search = np.zeros(dim)

            # Update position using exploration equation
            Positions[i] += (Positions[i] * levy_flight(levy_lambda)) - hazard_effect + local_search

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
