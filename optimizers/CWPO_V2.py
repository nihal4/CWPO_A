# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 00:00:00 2025

@author: Your Name
"""

import random
import numpy as np
import math
from solution import solution
import time

def CWPO(objf, lb, ub, dim, Population_size, Max_iter):

    # Ensure lb and ub are lists
    if isinstance(lb, (int, float)):
        lb = [lb] * dim
    if isinstance(ub, (int, float)):
        ub = [ub] * dim

    # Initialize population and parameters
    Cats = np.zeros((Population_size, dim))
    for i in range(dim):
        Cats[:, i] = (
            np.random.uniform(0, 1, Population_size) * (ub[i] - lb[i]) + lb[i]
        )

    EH = np.random.uniform(0.1, 1.0, (Population_size, dim))
    R = np.random.uniform(lb, ub, (Population_size, dim))

    Convergence_curve = np.zeros(Max_iter)
    s = solution()

    print('CWPO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for t in range(Max_iter):
        for i in range(Population_size):

            for j in range(dim):
                Cats[i, j] = np.clip(Cats[i, j], lb[j], ub[j])

            fitness = objf(Cats[i, :])

        EH = update_hazards(Cats, EH, t, alpha=1.0, beta=0.5, omega=0.1, p=2)

        for i in range(Population_size):
            Cats[i, :] = update_position(Cats[i, :], EH[i], R, dim, sf=1.0, lam=1.5)

        Convergence_curve[t] = np.min([objf(cat) for cat in Cats])

        if (t + 1) % 500 == 0:
            print(["At iteration " + str(t + 1) + " the best fitness is " + str(Convergence_curve[t])])

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "CWPO"
    s.bestIndividual = Cats[np.argmin([objf(cat) for cat in Cats])]
    s.objfname = objf.__name__

    return s

def update_hazards(Cats, EH, t, alpha, beta, omega, p):
    for i in range(len(Cats)):
        dynamic_hazard = alpha * np.linalg.norm(Cats[i] - random_hazard(t), ord=p)
        periodic_fluctuation = beta * math.cos(omega * t)
        EH[i] = dynamic_hazard + periodic_fluctuation
    return EH

def random_hazard(t):
    return np.array([math.sin(t), math.cos(t)])

def update_position(Cat, EH, R, dim, sf, lam):
    Levy = levy_flight(lam)
    Localized_search = np.sum(R / (np.linalg.norm(Cat - R, axis=1, keepdims=True) + 1e-8), axis=0)
    new_position = (Cat * Levy) - EH + Localized_search
    return new_position

def levy_flight(lam):
    """Generate a Levy flight step."""
    sigma = (np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2) /
             (np.math.gamma((1 + lam) / 2) * lam * 2 ** ((lam - 1) / 2))) ** (1 / lam)
    u = np.random.normal(0, sigma, 1)
    v = np.random.normal(0, 1, 1)
    step = u / abs(v) ** (1 / lam)
    return step[0]
