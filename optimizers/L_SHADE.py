import random
import numpy
import math
from solution import solution
import time

def L_SHADE(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    """
    LSHADE (Linear Success-History based Adaptive Differential Evolution) implementation
    
    Parameters:
    objf: objective function
    lb: lower bound
    ub: upper bound
    dim: dimension of the problem
    SearchAgents_no: initial population size
    Max_iter: maximum number of iterations
    """
    
    # Parameters
    memory_size = 5  # Size of the memory
    p_best_size = 5  # Number of best solutions considered for adaptation
    H = 6  # Scaling factor for population reduction
    
    # Initialize memory
    memory = [{"F": 0.5, "CR": 0.5} for _ in range(memory_size)]
    
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
        
    # Initialize population
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
    
    # Initialize fitness values
    fitness_values = numpy.zeros(SearchAgents_no)
    for i in range(SearchAgents_no):
        fitness_values[i] = objf(Positions[i, :])
    
    # Initialize best solution
    best_idx = numpy.argmin(fitness_values)
    best_solution = Positions[best_idx, :].copy()
    best_fitness = fitness_values[best_idx]
    
    # Initialize convergence curve
    Convergence_curve = numpy.zeros(Max_iter)
    s = solution()
    
    # Timer start
    print('LSHADE is optimizing  "' + objf.__name__ + '"')
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    # Main loop
    current_pop_size = SearchAgents_no
    for l in range(Max_iter):
        # Generate trial solutions
        trial_population = numpy.zeros((current_pop_size, dim))
        
        for i in range(current_pop_size):
            # Select parent indices
            parent_indices = random.sample(range(current_pop_size), 3)
            
            # Select scaling factor and crossover rate
            memory_indices = random.sample(range(memory_size), p_best_size)
            p_best_F = numpy.mean([memory[idx]["F"] for idx in memory_indices])
            p_best_CR = numpy.mean([memory[idx]["CR"] for idx in memory_indices])
            
            # Generate mutant vector with normal distribution
            F = max(0, numpy.random.normal(p_best_F, 0.1))
            CR = max(0, numpy.random.normal(p_best_CR, 0.1))
            
            # Mutation
            mutant_vector = Positions[parent_indices[0], :] + F * (
                Positions[parent_indices[1], :] - Positions[parent_indices[2], :])
            
            # Crossover
            j_rand = random.randint(0, dim-1)
            trial_population[i] = Positions[i, :].copy()
            
            for j in range(dim):
                if random.random() <= CR or j == j_rand:
                    trial_population[i, j] = mutant_vector[j]
                    
            # Boundary control
            for j in range(dim):
                trial_population[i, j] = numpy.clip(trial_population[i, j], lb[j], ub[j])
        
        # Evaluate trial solutions
        trial_fitness = numpy.zeros(current_pop_size)
        for i in range(current_pop_size):
            trial_fitness[i] = objf(trial_population[i, :])
            
            # Selection
            if trial_fitness[i] < fitness_values[i]:
                Positions[i, :] = trial_population[i, :].copy()
                fitness_values[i] = trial_fitness[i]
                
                if trial_fitness[i] < best_fitness:
                    best_fitness = trial_fitness[i]
                    best_solution = trial_population[i, :].copy()
        
        # Update memory
        sorted_indices = numpy.argsort(fitness_values)
        for i in range(min(memory_size, current_pop_size)):
            memory[i]["F"] += 0.1 * (random.random() - 0.5)
            memory[i]["CR"] += 0.1 * (random.random() - 0.5)
            
            # Ensure F and CR stay in [0,1]
            memory[i]["F"] = numpy.clip(memory[i]["F"], 0, 1)
            memory[i]["CR"] = numpy.clip(memory[i]["CR"], 0, 1)
        
        # Population reduction
        current_pop_size = max(2, round(current_pop_size - H * l / Max_iter))
        Positions = Positions[sorted_indices[:current_pop_size], :]
        fitness_values = fitness_values[sorted_indices[:current_pop_size]]
        
        # Update convergence curve
        Convergence_curve[l] = best_fitness
        
        if l % 1 == 0:
            print(["At iteration " + str(l) + " the best fitness is " + str(best_fitness)])
    
    # Timer end
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "LSHADE"
    s.bestIndividual = best_solution
    s.objfname = objf.__name__
    
    return s
