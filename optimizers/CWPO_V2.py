import numpy as np
import math
import time
from solution import solution

def levy_flight(lam, DIM, s=1.6):
    """Generate a Levy flight step for given lam and dimensionality DIM."""
    sigma = (math.gamma(1 + lam) * np.sin(np.pi * lam / 2) /
             (math.gamma((1 + lam) / 2) * lam * 2 ** ((lam - 1) / 2))) ** (1 / lam)
    
    # Generate Levy flight steps for each dimension
    u = np.random.normal(0, sigma, DIM)
    v = np.random.normal(0, 1, DIM)
    
    # Calculate the Levy flight step with constant s
    step = s * u / np.abs(v) ** (1 / lam)
    
    return step

def CWOP(objf, lb, ub, dim, SearchAgents_no, Max_iter, f=1.0, levy_lam=1.6, beta=1.0):

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize cats (search agents)
    cats = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        cats[:, i] = np.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]

    Convergence_curve = np.zeros(Max_iter)
    s = solution()

    # Loop counter
    print('CWPO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Main loop
    t = 1
    omega = 2 * np.pi * f

    diversity_threshold = 1e-3  # Threshold for population diversity

    while t <= Max_iter:
        # Calculate fitness of all cats
        cat_fitness = np.array([objf(ind) for ind in cats])

        # Find the best cat
        best_cat_index = np.argmin(cat_fitness)
        cat_best = cats[best_cat_index].copy()
        best_fitness = cat_fitness[best_cat_index]

        # Update positions of cats
        for i in range(SearchAgents_no):

            for j in range(dim):
                cats[i, j] = np.clip(cats[i, j], lb[j], ub[j])

            levy_step = levy_flight(levy_lam, dim)

            Catm = np.mean(cats, axis=0)

            alpha = np.random.rand() * beta

            if np.random.rand() <= 0.5:
                if cat_fitness[i] < best_fitness:
                    cats[i] = cat_best * (1 - t / Max_iter) + (Catm - cat_best) * np.random.rand()
                    best_fitness = cat_fitness[i]
                else:
                    cats[i] = alpha * cat_best - beta * np.cos(omega * t)

            else:
                catr = cats[np.random.randint(SearchAgents_no)]
                cats[i] = (cat_best * levy_step) + (catr * np.random.rand())

        # Monitor and maintain diversity
        population_diversity = np.mean(np.std(cats, axis=0))
        if population_diversity < diversity_threshold:
            # Reinitialize a fraction of the population
            reinit_fraction = 0.2  # Percentage of population to reinitialize
            num_reinit = max(1, int(SearchAgents_no * reinit_fraction))
            for i in range(num_reinit):
                cats[np.random.randint(SearchAgents_no)] = np.random.uniform(lb, ub, dim)

        # Record the best fitness
        Convergence_curve[t - 1] = best_fitness

        if t % 500 == 0:
            print(["At iteration " + str(t) + " the best fitness is " + str(best_fitness)])

        t += 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "CWPO"
    s.bestIndividual = cat_best
    s.objfname = objf.__name__

    return s

"""
import numpy as np
import math
import time
from solution import solution

def levy_flight(lam, DIM, s=0.01):
    #Generate a Levy flight step for given lam and dimensionality DIM.
    sigma = (math.gamma(1 + lam) * np.sin(np.pi * lam / 2) /
             (math.gamma((1 + lam) / 2) * lam * 2 ** ((lam - 1) / 2))) ** (1 / lam)
    
    # Generate Levy flight steps for each dimension
    u = np.random.normal(0, sigma, DIM)
    v = np.random.normal(0, 1, DIM)
    
    # Calculate the Levy flight step with constant s
    step = s * u / np.abs(v) ** (1 / lam)
    
    return step

def CWPO(objf, lb, ub, dim, SearchAgents_no, Max_iter, f, levy_lam, beta):

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize cats (search agents)
    cats = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        cats[:, i] = np.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]

    Convergence_curve = np.zeros(Max_iter)
    s = solution()

    # Loop counter
    print('CWPO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Main loop
    t = 1
    omega = 2 * np.pi * f

    while t <= Max_iter:
        # Calculate fitness of all cats
        cat_fitness = np.array([objf(ind) for ind in cats])

        # Find the best cat
        best_cat_index = np.argmin(cat_fitness)
        cat_best = cats[best_cat_index].copy()
        best_fitness = cat_fitness[best_cat_index]

        # Update positions of cats
        for i in range(SearchAgents_no):

            for j in range(dim):
                cats[i, j] = np.clip(cats[i, j], lb[j], ub[j])

            levy_step = levy_flight(levy_lam, dim)

            Catm = np.mean(cats, axis=0)

            alpha = np.random.rand() * beta

            if np.random.rand() <= 0.5:
                if cat_fitness[i] < best_fitness:
                    cats[i] = cat_best * (1 - t / Max_iter) + (Catm - cat_best) * np.random.rand()
                    best_fitness = cat_fitness[i]
                else:
                    cats[i] = alpha * cat_best - beta * np.cos(omega * t)

            else:
                catr = cats[np.random.randint(SearchAgents_no)]
                cats[i] = (cat_best * levy_step) + (catr * np.random.rand())

        # Record the best fitness
        Convergence_curve[t - 1] = best_fitness

        if t % 500 == 0:
            print(["At iteration " + str(t) + " the best fitness is " + str(best_fitness)])

        t += 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "CWPO"
    s.bestIndividual = cat_best
    s.objfname = objf.__name__

    return s
"""
