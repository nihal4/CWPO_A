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

    #new line
    lb = numpy.array(lb)
    ub = numpy.array(ub)
    
    # Initialize population
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
    
    # Initialize fitness values and arrays for single evaluations
    fitness_values = numpy.zeros(SearchAgents_no)
    for i in range(SearchAgents_no):
        fitness_values[i] = objf(Positions[i])
    
    # Initialize best solution
    best_idx = numpy.argmin(fitness_values)
    best_solution = Positions[best_idx].copy()
    best_fitness = fitness_values[best_idx]
    
    # Initialize convergence curve
    convergence = numpy.zeros(Max_iter)
    
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
            
            # Get parent vectors
            x_i = Positions[i]
            x_r1 = Positions[parent_indices[0]]
            x_r2 = Positions[parent_indices[1]]
            x_r3 = Positions[parent_indices[2]]
            
            # Mutation
            v_i = x_r1 + F * (x_r2 - x_r3)
            
            # Crossover
            j_rand = random.randint(0, dim-1)
            trial_vector = x_i.copy()
            
            for j in range(dim):
                if random.random() <= CR or j == j_rand:
                    trial_vector[j] = v_i[j]
            
            # Boundary control
            trial_vector = numpy.clip(trial_vector, lb, ub)
            trial_population[i] = trial_vector
        
        # Evaluate trial solutions and perform selection
        for i in range(current_pop_size):
            trial_fitness = objf(trial_population[i])
            
            if trial_fitness < fitness_values[i]:
                Positions[i] = trial_population[i].copy()
                fitness_values[i] = trial_fitness
                
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial_population[i].copy()
        
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
        convergence[l] = best_fitness
        
        if (l+1) % 500 == 0:
            #print(best_fitness)
            print("At iteration " + str(l+1) + " the best fitness is " + str(best_fitness))
    
    # Timer end
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = numpy.array(convergence)
    s.optimizer = "LSHADE"
    s.bestIndividual = best_solution
    s.objfname = objf.__name__
    
    return s
