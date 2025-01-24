# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 2025

@author: Stray Cat Researcher
"""

import random
import numpy as np
import math
from solution import solution
import time

def CWPO(objf, lb, ub, dim, CatAgents_no, Max_iter):

    # Initialize population and parameters
    Best_pos = np.zeros(dim)
    Best_score = float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize the positions of cat agents
    Positions = np.zeros((CatAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (
            np.random.uniform(0, 1, CatAgents_no) * (ub[i] - lb[i]) + lb[i]
        )

    Convergence_curve = np.zeros(Max_iter)
    s = solution()

    # Initialize hazard and resource matrices
    Hazards = np.random.uniform(0, 1, (CatAgents_no, dim))
    Resources = np.random.uniform(0, 1, (CatAgents_no, dim))

    print('CWPO is optimizing "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Main loop
    for t in range(0, Max_iter):
        for i in range(0, CatAgents_no):
            # Ensure agents remain within bounds
            for j in range(dim):
                Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[j])

            # Calculate fitness for each agent
            fitness = objf(Positions[i, :])

            # Update best solution
            if fitness < Best_score:
                Best_score = fitness
                Best_pos = Positions[i, :].copy()

        # Update positions based on local and global behavior
        for i in range(0, CatAgents_no):
            # Local exploration and exploitation
            Local_move = np.zeros(dim)
            for j in range(dim):
                nearby_resources = Resources[np.random.randint(0, CatAgents_no), :]
                dist = np.linalg.norm(Positions[i, :] - nearby_resources)
                if dist > 0:
                    Local_move[j] = (
                        (nearby_resources[j] - Positions[i, j]) / dist
                    )

            # Hazard avoidance
            Hazard_move = np.zeros(dim)
            for j in range(dim):
                current_hazard = Hazards[i, :]
                hazard_dist = np.linalg.norm(Positions[i, :] - current_hazard)
                if hazard_dist > 0:
                    Hazard_move[j] = -0.5 * (current_hazard[j] - Positions[i, j]) / hazard_dist

            # Global exploration using Lévy flight
            Levy_jump = np.zeros(dim)
            for j in range(dim):
                Levy_jump[j] = (
                    0.01
                    * np.sign(random.random() - 0.5)
                    * np.abs(np.random.normal(0, 1)) ** (1 / 1.5)
                )

            # Update position
            Positions[i, :] += (
                0.5 * Local_move
                + 0.3 * Hazard_move
                + 0.2 * Levy_jump
            )

        Convergence_curve[t] = Best_score

        if (t + 1) % 500 == 0:
            print(["At iteration " + str(t + 1) + " the best fitness is " + str(Best_score)])

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "CWPO"
    s.bestIndividual = Best_pos
    s.objfname = objf.__name__

    return s
