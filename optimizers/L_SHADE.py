import random
import numpy
import math
from solution import solution
import time

def L_SHADE(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    """
    LSHADE (Linear Success-History based Adaptive Differential Evolution) implementation
    """
    
    # Initialize solution object first
    s = solution()
    
    # Parameters
    memory_size = 5
    p_best_size = 5
    H = 6
    
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
    fitness_values = numpy.array([objf(Positions[i, :]) for i in range(SearchAgents_no)])
    
    # Initialize best solution
    best_idx = numpy.argmin(fitness_values)
    best_solution = Positions[best_idx, :].copy()
    best_fitness = fitness_values[best_idx]
    
    # Initialize convergence curve
    convergence = []
    
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
            if current_pop_size < 4:
                parent_indices = [i]
                while len(parent_indices) < 3:
                    idx = random.randint(0, current_pop_size - 1)
                    if idx not in parent_indices:
                        parent_indices.append(idx)
            else:
                parent_indices = random.sample(range(current_pop_size), 3)
            
            # Select scaling factor and crossover rate
            memory_indices = random.sample(range(min(memory_size, current_pop_size)), 
                                        min(p_best_size, min(memory_size, current_pop_size)))
            p_best_F = numpy.mean([memory[idx]["F"] for idx in memory_indices])
            p_best_CR = numpy.mean([memory[idx]["CR"] for idx in memory_indices])
            
            # Generate parameters
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
        trial_fitness = numpy.array([objf(trial_population[i, :]) for i in range(current_pop_size)])
            
        # Selection
        improved_indices = trial_fitness < fitness_values
        #Positions[improved_indices] = trial_population[improved_indices]
        Positions[improved_indices, :] = trial_population[improved_indices, :] #2d problem
        fitness_values[improved_indices] = trial_fitness[improved_indices]
        
        # Update best solution
        current_best_idx = numpy.argmin(fitness_values)
        if fitness_values[current_best_idx] < best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_solution = Positions[current_best_idx, :].copy()
        
        # Update memory
        sorted_indices = numpy.argsort(fitness_values)
        for i in range(min(memory_size, current_pop_size)):
            memory[i]["F"] = numpy.clip(memory[i]["F"] + 0.1 * (random.random() - 0.5), 0, 1)
            memory[i]["CR"] = numpy.clip(memory[i]["CR"] + 0.1 * (random.random() - 0.5), 0, 1)
        
        # Population reduction
        new_pop_size = max(4, round(SearchAgents_no - (SearchAgents_no - 4) * l / Max_iter))
        if new_pop_size < current_pop_size:
            current_pop_size = new_pop_size
            Positions = Positions[sorted_indices[:current_pop_size]]
            fitness_values = fitness_values[sorted_indices[:current_pop_size]]
        
        # Store convergence data
        convergence.append(best_fitness)
        
        if (l+1) % 500 == 0:
            print(["At iteration " + str(l+1) + " the best fitness is " + str(best_fitness)])
    
    # Timer end
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = numpy.array(convergence)
    s.optimizer = "LSHADE"
    s.bestIndividual = best_solution
    s.objfname = objf.__name__
    
    return s
