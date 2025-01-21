import numpy
import random
import math
from solution import solution
import time

def L_SHADE(objf, lb, ub, dim, SearchAgents_no, Max_iter):

    # initialize the population
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(SearchAgents_no):
        Positions[i, :] = numpy.random.uniform(lb, ub, dim)

    # Historical solutions for adaptation in mutation
    BestSolutions = numpy.zeros((SearchAgents_no, dim))
    Convergence_curve = numpy.zeros(Max_iter)
    s = solution()

    # Loop counter
    print('L_SHADE is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Main loop
    for l in range(Max_iter):
        for i in range(SearchAgents_no):

            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])

            # Calculate the objective function for each agent
            fitness = objf(Positions[i, :])

            # Update the historical best solution (elitism)
            if fitness < BestSolutions[i, :].fitness:
                BestSolutions[i, :] = Positions[i, :]

        # Generate new candidates using mutation strategy (DE/rand/1/bin or any variation)
        for i in range(SearchAgents_no):
            # Mutation using DE/rand/1 strategy
            r1, r2, r3 = random.sample(range(SearchAgents_no), 3)
            mutant_vector = Positions[r1, :] + 0.8 * (Positions[r2, :] - Positions[r3, :])

            # Ensure bounds are respected
            mutant_vector = numpy.clip(mutant_vector, lb, ub)

            # Crossover operation (binomial)
            trial_vector = numpy.copy(Positions[i, :])
            for j in range(dim):
                if random.random() < 0.5:
                    trial_vector[j] = mutant_vector[j]

            # Evaluate the trial solution
            trial_fitness = objf(trial_vector)

            # Selection: if the trial solution is better, accept it
            if trial_fitness < fitness:
                Positions[i, :] = trial_vector
                BestSolutions[i, :] = trial_vector

        # Track the best solution
        Convergence_curve[l] = min([objf(Positions[i, :]) for i in range(SearchAgents_no)])

        if l % 1 == 0:
            print(f"At iteration {l} the best fitness is {Convergence_curve[l]}")

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "L_SHADE"
    s.bestIndividual = BestSolutions[numpy.argmin(Convergence_curve)]
    s.objfname = objf.__name__

    return s
