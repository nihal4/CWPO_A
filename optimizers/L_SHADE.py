import numpy as np
import random
import time
from solution import solution

def L_SHADE(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    # Parameters
    memory_size = 5  # Size of the memory
    p_best_size = 5  # Number of best solutions considered for adaptation
    H = 6  # Scaling factor for population reduction

    # Initialize population
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    population = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (np.array(ub) - np.array(lb)) + np.array(lb)
    fitness_values = np.array([objf(ind) for ind in population])

    # Initialize best solution and fitness
    best_index = np.argmin(fitness_values)
    best_fitness = fitness_values[best_index]
    best_solution = population[best_index].copy()

    # Initialize memory
    memory = [{"F": 0.5, "CR": 0.5} for _ in range(memory_size)]

    # Initialize solution object
    s = solution()
    print(f"LSHADE is optimizing '{objf.__name__}'")
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Main loop
    for iter in range(Max_iter):
        # Generate trial solutions
        trial_population = np.zeros((SearchAgents_no, dim))
        for i in range(SearchAgents_no):
            # Select parent indices
            parent_indices = np.random.choice(SearchAgents_no, 3, replace=False)

            # Select scaling factor and crossover rate
            sample_size = min(memory_size, p_best_size)
            if sample_size <= len(range(memory_size)):
                memory_indices = random.sample(range(memory_size), sample_size)
            else:
                memory_indices = random.sample(range(memory_size), memory_size)
                
            p_best_F = np.mean([memory[idx]["F"] for idx in memory_indices])
            p_best_CR = np.mean([memory[idx]["CR"] for idx in memory_indices])


            # Generate mutant vector
            F = max(0, np.random.normal(p_best_F, 0.1))
            CR = max(0, np.random.normal(p_best_CR, 0.1))
            mutant_vector = population[parent_indices[0]] + F * (population[parent_indices[1]] - population[parent_indices[2]])

            # Crossover
            j_rand = random.randint(0, dim - 1)
            trial_population[i] = population[i].copy()
            for j in range(dim):
                if random.random() <= CR or j == j_rand:
                    trial_population[i, j] = mutant_vector[j]

        # Evaluate trial solutions
        trial_fitness_values = np.array([objf(ind) for ind in trial_population])

        # Update population
        for i in range(SearchAgents_no):
            if trial_fitness_values[i] < fitness_values[i]:
                population[i] = trial_population[i]
                fitness_values[i] = trial_fitness_values[i]
                if trial_fitness_values[i] < best_fitness:
                    best_fitness = trial_fitness_values[i]
                    best_solution = trial_population[i].copy()

        # Update memory
        sorted_indices = np.argsort(fitness_values)
        for i in range(min(memory_size, SearchAgents_no)):
            memory[i]["F"] += 0.1 * (random.random() - 0.5)
            memory[i]["CR"] += 0.1 * (random.random() - 0.5)

        # Population reduction
        new_population_size = max(2, round(SearchAgents_no - H * iter / Max_iter))
        population = population[sorted_indices[:new_population_size]]
        fitness_values = fitness_values[sorted_indices[:new_population_size]]
        SearchAgents_no = new_population_size

        # Track convergence
        if iter % 1 == 0:
            print(f"At iteration {iter}, the best fitness is {best_fitness}")

    # Finalize solution
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = fitness_values.tolist()
    s.optimizer = "LSHADE"
    s.bestIndividual = best_solution
    s.objfname = objf.__name__

    return s
