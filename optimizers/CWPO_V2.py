# -*- coding: utf-8 -*-
"""
Cat Water Phobia Optimizer (CWPO)
Inspired by the survival strategies of stray cats
"""

import numpy as np
import time
from solution import solution

def dynamic_hazard(t, omega, alpha, beta, p, cat_pos, hazard_pos, iteration, max_iter):
    """Adaptive environmental hazard."""
    distance = np.linalg.norm(cat_pos - hazard_pos) ** p
    periodic_fluctuation = beta * np.cos(omega * t)
    
    # Adjust hazard effect based on iteration
    adaptive_alpha = alpha * (1 - iteration / max_iter)
    adaptive_beta = beta * (1 - iteration / max_iter)

    return adaptive_alpha * distance + periodic_fluctuation


def levy_flight_adaptive(levy_lambda, iteration, max_iter):
    """Adaptive Levy flight."""
    sigma = (np.math.gamma(1 + levy_lambda) * np.sin(np.pi * levy_lambda / 2) /
             (np.math.gamma((1 + levy_lambda) / 2) * levy_lambda * 2 ** ((levy_lambda - 1) / 2))) ** (1 / levy_lambda)
    u = np.random.normal(0, sigma, 1)
    v = np.random.normal(0, 1, 1)
    step = u / abs(v) ** (1 / levy_lambda)
    
    # Adapt the step size based on iteration (reduce step size as the algorithm progresses)
    adaptive_step = step[0] * (1 - iteration / max_iter)
    return adaptive_step


def CWPO(objf, lb, ub, dim, SearchAgents_no, Max_iter, alpha=1.5, beta=0.5, omega_freq=0.1, sf=5.0, levy_lambda=1.5, p=-1):
    omega = 2 * np.pi * omega_freq

    Positions = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb
    Hazards = np.random.uniform(lb, ub, (SearchAgents_no, dim))
    Resources = np.random.uniform(lb, ub, (SearchAgents_no, dim))

    Alpha_pos = np.zeros(dim)
    Alpha_score = float("inf")

    Convergence_curve = np.zeros(Max_iter)
    s = solution()

    print('CWPO is optimizing "' + objf.__name__ + '"')
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(Max_iter):
        for i in range(SearchAgents_no):
            # Dynamic environmental hazard with adaptive parameters
            hazard_effect = dynamic_hazard(
                t=l,
                omega=omega,
                alpha=alpha,
                beta=beta,
                p=p,
                cat_pos=Positions[i],
                hazard_pos=Hazards[i],
                iteration=l,
                max_iter=Max_iter
            )

            distances_to_resources = np.linalg.norm(Resources - Positions[i], axis=1)
            nearby_resources = Resources[distances_to_resources <= sf]
            local_search = nearby_resources.mean(axis=0) if nearby_resources.size > 0 else np.zeros(dim)

            # Update position using adaptive Levy flight and hazard effect
            step = levy_flight_adaptive(levy_lambda, l, Max_iter)
            Positions[i] += (Positions[i] * step) - hazard_effect + local_search

            # Ensure position stays within bounds
            Positions[i] = np.clip(Positions[i], lb, ub)

            # Calculate fitness
            fitness = objf(Positions[i])

            if fitness < Alpha_score:
                Alpha_score = fitness
                Alpha_pos = Positions[i].copy()

        Convergence_curve[l] = Alpha_score

        if (l+1) % 500 == 0:
            print(f"At iteration {l + 1}, the best fitness is {Alpha_score}")

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "CWPO"
    s.bestIndividual = Alpha_pos
    s.objfname = objf.__name__

    return s
