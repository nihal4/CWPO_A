import numpy as np
import random
import time
from solution import solution

def GGO(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    # Initialize population
    population = np.random.uniform(lb, ub, (SearchAgents_no, dim))
    fitness = np.zeros(SearchAgents_no)

    # Evaluate fitness of each agent
    for i in range(SearchAgents_no):
        fitness[i] = objf(population[i, :])

    # Initialize variables for best solution
    best_solution = np.zeros(dim)
    best_fitness = float("inf")
    s = solution()

    # Timer start
    print('GGO is optimizing "' + objf.__name__ + '"')
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Main loop
    for iter in range(Max_iter):
        for i in range(SearchAgents_no):
            # Determine the best agent in the population
            best_agent_index = np.argmin(fitness)

            # Generate a new solution by combining exploration and exploitation
            new_solution = population[i, :] + np.random.rand(dim) * (population[best_agent_index, :] - population[i, :])

            # Clip new solution to ensure it stays within bounds
            new_solution = np.clip(new_solution, lb, ub)

            # Evaluate fitness of the new solution
            new_fitness = objf(new_solution)

            # Update if the new solution is better
            if new_fitness < fitness[i]:
                population[i, :] = new_solution
                fitness[i] = new_fitness

        # Track the best solution found so far
        best_fitness_iter = np.min(fitness)
        if best_fitness_iter < best_fitness:
            best_fitness = best_fitness_iter
            best_solution = population[np.argmin(fitness), :]

        print(f"Iteration {iter + 1}: Best fitness = {best_fitness}")

    # Timer end
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = fitness
    s.optimizer = "GGO"
    s.bestIndividual = best_solution
    s.objfname = objf.__name__

    return s
