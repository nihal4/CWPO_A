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

def dynamic_hazard(cat_pos, objf):
    """Calculate the hazard at the current position (based on fitness)."""
    hazard = objf(cat_pos)
    return hazard

def CWPO(objf, lb, ub, dim, SearchAgents_no, Max_iter, levy_lambda=1.5, sf=5.0):
    """
    Cat Water Phobia Optimizer (CWPO)
    objf: Objective function
    lb: Lower bound
    ub: Upper bound
    dim: Number of dimensions
    SearchAgents_no: Population size
    Max_iter: Maximum iterations
    levy_lambda: Levy flight parameter
    sf: Search radius (for local exploration)
    """
    
    # Initialize positions randomly
    Positions = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb

    # Initialize the best solution
    Alpha_pos = np.zeros(dim)
    Alpha_score = float("inf")

    Convergence_curve = np.zeros(Max_iter)
    s = solution()

    print('CWPO is optimizing "' + objf.__name__ + '"')
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(Max_iter):
        for i in range(SearchAgents_no):
            # Store previous position to revert if needed
            previous_position = Positions[i].copy()
            previous_fitness = dynamic_hazard(previous_position, objf)
            
            # Calculate the hazard at the current position (based on objective function)
            hazard_effect = dynamic_hazard(Positions[i], objf)
            
            # Calculate resources based on hazard level (inverse proportionality)
            resources = 1 / (1 + hazard_effect)  # More resources for lower hazard
            
            # Local exploration: Look for nearby resources (local minima or low hazard areas)
            distances_to_resources = np.linalg.norm(Positions - Positions[i], axis=1)
            nearby_resources = Positions[distances_to_resources <= sf]
            
            # Average the positions of nearby resources for localized exploration
            local_search = nearby_resources.mean(axis=0) if nearby_resources.size > 0 else np.zeros(dim)
            
            # Update position using Levy flight with exploration and local search
            step = levy_flight(levy_lambda)
            new_position = (Positions[i] * step) - hazard_effect + local_search
            
            # Calculate the hazard of the new position
            new_hazard = dynamic_hazard(new_position, objf)
            
            # If the new position has higher hazard, revert to the previous position and explore in a new direction
            if new_hazard > previous_fitness:
                Positions[i] = previous_position
                # Explore in a new direction (e.g., apply a different Levy flight step)
                new_position = (Positions[i] * levy_flight(levy_lambda)) - hazard_effect + local_search
                Positions[i] = np.clip(new_position, lb, ub)
            else:
                # Otherwise, update the position
                Positions[i] = np.clip(new_position, lb, ub)

            # Calculate fitness
            fitness = dynamic_hazard(Positions[i], objf)

            # Update the best solution (global minimum)
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
