import numpy as np
import random
import time
from solution import solution

def L_SHADE(objf, lb, ub, dim, max_iter, population_size):
    """
    LSHADE Algorithm Implementation in Python

    Parameters:
        objf: Objective function to minimize
        lb: Lower boundary (scalar or list)
        ub: Upper boundary (scalar or list)
        dim: Dimensionality of the problem
        max_iter: Maximum number of iterations
        population_size: Initial population size

    Returns:
        A solution object containing optimization results
    """
    # Parameters
    memory_size = 5  # Size of the memory
    p_best_size = 5  # Number of best solutions considered for adaptation
    H = 6  # Scaling factor for population reduction

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize population
    population = np.random.rand(population_size, dim) * (np.array(ub) - np.array(lb)) + np.array(lb)
    fitness_values = np.apply_along_axis(objf, 1, population)

    # Initialize best solution and fitness
    best_index = np.argmin(fitness_values)
    best_fitness = fitness_values[best_index]
    best_solution = population[best_index, :].copy()

    # Initialize memory
    memory = [{'F': 0.5, 'CR': 0.5} for _ in range(memory_size)]

    # Convergence curve
    convergence_curve = np.zeros(max_iter)
    s = solution()

    print(f'LSHADE is optimizing "{objf.__name__}"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Main loop
    for iter in range(max_iter):
        trial_population = np.zeros_like(population)

        for i in range(population_size):
            # Select parent indices
            parent_indices = random.sample(range(population_size), 3)

            # Select scaling factor and crossover rate
            memory_indices = random.sample(range(memory_size), p_best_size)
            p_best_F = np.mean([memory[idx]['F'] for idx in memory_indices])
            p_best_CR = np.mean([memory[idx]['CR'] for idx in memory_indices])

            # Generate mutant vector
            F = max(0, np.random.normal(p_best_F, 0.1))
            CR = max(0, np.random.normal(p_best_CR, 0.1))
            mutant_vector = population[parent_indices[0]] + F * (population[parent_indices[1]] - population[parent_indices[2]])

            # Crossover
            j_rand = random.randint(0, dim - 1)
            trial_vector = population[i, :].copy()
            for j in range(dim):
                if random.random() <= CR or j == j_rand:
                    trial_vector[j] = mutant_vector[j]

            trial_population[i, :] = trial_vector

        # Evaluate trial solutions
        trial_fitness_values = np.apply_along_axis(objf, 1, trial_population)

        # Update population
        for i in range(population_size):
            if trial_fitness_values[i] < fitness_values[i]:
                population[i, :] = trial_population[i, :]
                fitness_values[i] = trial_fitness_values[i]

                if trial_fitness_values[i] < best_fitness:
                    best_fitness = trial_fitness_values[i]
                    best_solution = trial_population[i, :].copy()

        # Update memory
        sorted_indices = np.argsort(fitness_values)
        for idx in range(min(memory_size, population_size)):
            memory[idx]['F'] += 0.1 * (random.random() - 0.5)
            memory[idx]['CR'] += 0.1 * (random.random() - 0.5)

        # Record convergence
        convergence_curve[iter] = best_fitness

        # Print progress
        if iter % 1 == 0:
            print(f"At iteration {iter} the best fitness is {best_fitness}")

        # Population reduction
        reduced_size = max(2, int(population_size - H * iter / max_iter))
        population = population[sorted_indices[:reduced_size], :]
        fitness_values = fitness_values[sorted_indices[:reduced_size]]
        population_size = reduced_size

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "LSHADE"
    s.bestIndividual = best_solution
    s.objfname = objf.__name__

    return s
