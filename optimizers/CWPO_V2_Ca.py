# -*- coding: utf-8 -*-
"""
Cat Water Phobia Optimizer (CWPO) with Historical Memory, Resource Redistribution, Multi-Population Mechanism, Elite Reinforcement, Fitness-Based Hazard Adaptation, and Dynamic Parameter Adaptation
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

def dynamic_hazard(t, omega, alpha, beta, p, cat_pos, hazard_pos, fitness, max_fitness):
    """
    Calculate the environmental hazard at time t with fitness-based adaptation.
    EH_i = (alpha * ||C_i - O(t)||^p + beta * cos(omega * t)) * (1 - fitness / max_fitness)
    """
    distance = np.linalg.norm(cat_pos - hazard_pos) ** p
    periodic_fluctuation = beta * np.cos(omega * t)
    hazard_intensity = alpha * distance + periodic_fluctuation
    adaptation_factor = 1 - (fitness / max_fitness)
    return hazard_intensity * adaptation_factor

def adapt_parameters(iteration, max_iter, alpha, beta, omega_freq, sf, levy_lambda):
    """
    Dynamically adapt parameters over iterations.
    """
    alpha *= (1 - iteration / max_iter)  # Decrease alpha over time
    beta *= (1 - iteration / max_iter)   # Decrease beta over time
    omega_freq *= (1 + iteration / max_iter * 0.5)  # Increase omega_freq slightly
    sf *= (1 - iteration / max_iter)  # Decrease search radius
    levy_lambda *= (1 - iteration / max_iter * 0.5)  # Decrease levy_lambda for smaller steps
    return alpha, beta, omega_freq, sf, levy_lambda

def CWPO(objf, lb, ub, dim, SearchAgents_no, Max_iter, alpha=0.1, beta=0.5, omega_freq=0.1, sf=5.0, levy_lambda=1.7, p=-1.5, resource_update_interval=10, num_subgroups=6, exchange_interval=10, elite_reinforcement_interval=5, elite_influence=0.7):
    """
    Cat Water Phobia Optimizer (CWPO) with Historical Memory, Resource Redistribution, Multi-Population Mechanism, Elite Reinforcement, Fitness-Based Hazard Adaptation, and Dynamic Parameter Adaptation
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
    resource_update_interval: Number of iterations after which resources are redistributed
    num_subgroups: Number of subgroups for multi-population mechanism
    exchange_interval: Number of iterations after which subgroups exchange information
    elite_reinforcement_interval: Number of iterations after which elite reinforcement is applied
    elite_influence: Factor controlling the influence of the global best (Alpha_pos) on other agents
    """

    # Calculate angular frequency
    omega = 2 * np.pi * omega_freq

    # Initialize population, hazards, and resources
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = np.random.uniform(0, 1, SearchAgents_no) * (ub - lb) + lb

    Hazards = np.random.uniform(lb, ub, (SearchAgents_no, dim))
    Resources = np.random.uniform(lb, ub, (SearchAgents_no, dim))

    # Initialize historical memory (personal bests)
    Personal_best = Positions.copy()
    Personal_best_scores = np.full(SearchAgents_no, float("inf"))

    # Initialize the best solution globally
    Alpha_pos = np.zeros(dim)
    Alpha_score = float("inf")

    Convergence_curve = np.zeros(Max_iter)
    s = solution()

    # Divide population into subgroups
    subgroup_size = SearchAgents_no // num_subgroups
    subgroups = [list(range(i * subgroup_size, (i + 1) * subgroup_size)) for i in range(num_subgroups)]

    # Start optimization
    print('CWPO is optimizing "' + objf.__name__ + '"')
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(Max_iter):
        # Dynamically adapt parameters
        alpha, beta, omega_freq, sf, levy_lambda = adapt_parameters(l, Max_iter, alpha, beta, omega_freq, sf, levy_lambda)
        omega = 2 * np.pi * omega_freq

        # Redistribute resources periodically
        if l % resource_update_interval == 0:
            Resources = np.random.uniform(lb, ub, (SearchAgents_no, dim))

        max_fitness = max(Personal_best_scores) if max(Personal_best_scores) != float("inf") else 1

        for subgroup in subgroups:
            for i in subgroup:
                # Dynamic environmental hazard with fitness-based adaptation
                hazard_effect = dynamic_hazard(
                    t=l,
                    omega=omega,
                    alpha=alpha,
                    beta=beta,
                    p=p,
                    cat_pos=Positions[i],
                    hazard_pos=Hazards[i],
                    fitness=Personal_best_scores[i],
                    max_fitness=max_fitness
                )

                # Localized search
                distances_to_resources = np.linalg.norm(Resources - Positions[i], axis=1)
                nearby_resources = Resources[distances_to_resources <= sf]
                if nearby_resources.size > 0:
                    local_search = nearby_resources.mean(axis=0)
                else:
                    local_search = np.zeros(dim)

                # Update position using exploration equation and historical memory
                inertia = Positions[i] * levy_flight(levy_lambda)
                memory_influence = (Personal_best[i] - Positions[i]) * np.random.random()
                Positions[i] += inertia + memory_influence - hazard_effect + local_search

                # Ensure position stays within bounds
                Positions[i] = np.clip(Positions[i], lb, ub)

                # Calculate fitness
                fitness = objf(Positions[i])

                # Update personal bests
                if fitness < Personal_best_scores[i]:
                    Personal_best_scores[i] = fitness
                    Personal_best[i] = Positions[i].copy()

                # Update the global best solution
                if fitness < Alpha_score:
                    Alpha_score = fitness
                    Alpha_pos = Positions[i].copy()

        # Exchange information among subgroups periodically
        if l % exchange_interval == 0:
            for subgroup in subgroups:
                best_in_subgroup = min(subgroup, key=lambda idx: Personal_best_scores[idx])
                for other_subgroup in subgroups:
                    if subgroup != other_subgroup:
                        for idx in other_subgroup:
                            if Personal_best_scores[best_in_subgroup] < Personal_best_scores[idx]:
                                Personal_best[idx] = Personal_best[best_in_subgroup].copy()
                                Personal_best_scores[idx] = Personal_best_scores[best_in_subgroup]

        # Apply elite reinforcement periodically
        if l % elite_reinforcement_interval == 0:
            for i in range(SearchAgents_no):
                elite_influence_vector = elite_influence * (Alpha_pos - Positions[i]) * np.random.random()
                Positions[i] += elite_influence_vector
                Positions[i] = np.clip(Positions[i], lb, ub)

        Convergence_curve[l] = Alpha_score

        if (l+1) % 500 == 0:
            print(f"At iteration {l + 1}, the best fitness is {Alpha_score}")

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "CWPO_V3"
    s.bestIndividual = Alpha_pos
    s.objfname = objf.__name__

    return s


"""
import numpy as np
import random
import time
from solution import solution

def levy_flight(lam):
    #Generate a Levy flight step.
    sigma = (np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2) /
             (np.math.gamma((1 + lam) / 2) * lam * 2 ** ((lam - 1) / 2))) ** (1 / lam)
    u = np.random.normal(0, sigma, 1)
    v = np.random.normal(0, 1, 1)
    step = u / abs(v) ** (1 / lam)
    return step[0]

def dynamic_hazard(t, omega, alpha, beta, p, cat_pos, hazard_pos, fitness, max_fitness):
    
    #Calculate the environmental hazard at time t with fitness-based adaptation.
    #EH_i = (alpha * ||C_i - O(t)||^p + beta * cos(omega * t)) * (1 - fitness / max_fitness)
    
    distance = np.linalg.norm(cat_pos - hazard_pos) ** p
    periodic_fluctuation = beta * np.cos(omega * t)
    hazard_intensity = alpha * distance + periodic_fluctuation
    adaptation_factor = 1 - (fitness / max_fitness)
    return hazard_intensity * adaptation_factor

#best_parameter
#objf, lb, ub, dim, SearchAgents_no, Max_iter, alpha=0.1, beta=0.5, omega_freq=0.1, sf=5.0, levy_lambda=1.7, p=-1.5, resource_update_interval=10, num_subgroups=6, exchange_interval=25, elite_reinforcement_interval=5, elite_influence=0.7 
def CWPO(objf, lb, ub, dim, SearchAgents_no, Max_iter, alpha=0.1, beta=0.5, omega_freq=0.1, sf=5.0, levy_lambda=1.7, p=-1.5, resource_update_interval=10, num_subgroups=6, exchange_interval=10, elite_reinforcement_interval=5, elite_influence=0.7):
    
    #Cat Water Phobia Optimizer (CWPO) with Historical Memory, Resource Redistribution, Multi-Population Mechanism, Elite Reinforcement, and Fitness-Based Hazard Adaptation
    #objf: Objective function
    #lb: Lower bound
    #ub: Upper bound
    #dim: Number of dimensions
    #SearchAgents_no: Population size
    #Max_iter: Maximum iterations
    #alpha: Hazard impact scaling factor
    #beta: Hazard fluctuation factor
    #omega_freq: Frequency of hazard oscillation
    #sf: Search radius
    #levy_lambda: Levy flight parameter
    #p: Hazard decay exponent
    #resource_update_interval: Number of iterations after which resources are redistributed
    #num_subgroups: Number of subgroups for multi-population mechanism
    #exchange_interval: Number of iterations after which subgroups exchange information
    #elite_reinforcement_interval: Number of iterations after which elite reinforcement is applied
    #elite_influence: Factor controlling the influence of the global best (Alpha_pos) on other agents
    

    # Calculate angular frequency
    omega = 2 * np.pi * omega_freq

    # Ensure bounds are arrays of correct shape
    ub = np.array(ub)
    lb = np.array(lb)

    if ub.shape == () and lb.shape == ():  # Scalars
        Positions = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb
    elif ub.shape[0] == dim and lb.shape[0] == dim:  # Arrays of length `dim`
        Positions = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb
    else:
        raise ValueError("Bounds `ub` and `lb` must be either scalars or arrays of the same length as `dim")

    Hazards = np.random.uniform(lb, ub, (SearchAgents_no, dim))
    Resources = np.random.uniform(lb, ub, (SearchAgents_no, dim))

    # Initialize historical memory (personal bests)
    Personal_best = Positions.copy()
    Personal_best_scores = np.full(SearchAgents_no, float("inf"))

    # Initialize the best solution globally
    Alpha_pos = np.zeros(dim)
    Alpha_score = float("inf")

    Convergence_curve = np.zeros(Max_iter)
    s = solution()

    # Divide population into subgroups
    subgroup_size = SearchAgents_no // num_subgroups
    subgroups = [list(range(i * subgroup_size, (i + 1) * subgroup_size)) for i in range(num_subgroups)]

    # Start optimization
    print('CWPO is optimizing "' + objf.__name__ + '"')
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(Max_iter):
        # Redistribute resources periodically
        if l % resource_update_interval == 0:
            Resources = np.random.uniform(lb, ub, (SearchAgents_no, dim))

        max_fitness = max(Personal_best_scores) if max(Personal_best_scores) != float("inf") else 1

        for subgroup in subgroups:
            for i in subgroup:
                # Dynamic environmental hazard with fitness-based adaptation
                hazard_effect = dynamic_hazard(
                    t=l,
                    omega=omega,
                    alpha=alpha,
                    beta=beta,
                    p=p,
                    cat_pos=Positions[i],
                    hazard_pos=Hazards[i],
                    fitness=Personal_best_scores[i],
                    max_fitness=max_fitness
                )

                # Localized search
                distances_to_resources = np.linalg.norm(Resources - Positions[i], axis=1)
                nearby_resources = Resources[distances_to_resources <= sf]
                if nearby_resources.size > 0:
                    local_search = nearby_resources.mean(axis=0)
                else:
                    local_search = np.zeros(dim)

                # Update position using exploration equation and historical memory
                inertia = Positions[i] * levy_flight(levy_lambda)
                memory_influence = (Personal_best[i] - Positions[i]) * np.random.random()
                Positions[i] += inertia + memory_influence - hazard_effect + local_search

                # Ensure position stays within bounds
                Positions[i] = np.clip(Positions[i], lb, ub)

                # Calculate fitness
                fitness = objf(Positions[i])

                # Update personal bests
                if fitness < Personal_best_scores[i]:
                    Personal_best_scores[i] = fitness
                    Personal_best[i] = Positions[i].copy()

                # Update the global best solution
                if fitness < Alpha_score:
                    Alpha_score = fitness
                    Alpha_pos = Positions[i].copy()

        # Exchange information among subgroups periodically
        if l % exchange_interval == 0:
            for subgroup in subgroups:
                best_in_subgroup = min(subgroup, key=lambda idx: Personal_best_scores[idx])
                for other_subgroup in subgroups:
                    if subgroup != other_subgroup:
                        for idx in other_subgroup:
                            if Personal_best_scores[best_in_subgroup] < Personal_best_scores[idx]:
                                Personal_best[idx] = Personal_best[best_in_subgroup].copy()
                                Personal_best_scores[idx] = Personal_best_scores[best_in_subgroup]

        # Apply elite reinforcement periodically
        if l % elite_reinforcement_interval == 0:
            for i in range(SearchAgents_no):
                elite_influence_vector = elite_influence * (Alpha_pos - Positions[i]) * np.random.random()
                Positions[i] += elite_influence_vector
                Positions[i] = np.clip(Positions[i], lb, ub)

        Convergence_curve[l] = Alpha_score

        if (l+1) % 500 == 0:
            print(f"At iteration {l + 1}, the best fitness is {Alpha_score}")

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "CWPO with Historical Memory, Resource Redistribution, Multi-Population Mechanism, Elite Reinforcement, and Fitness-Based Hazard Adaptation"
    s.bestIndividual = Alpha_pos
    s.objfname = objf.__name__

    return s
"""
