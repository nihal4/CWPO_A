import numpy as np
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


def CWPO(objf, lb, ub, dim, SearchAgents_no, Max_iter, levy_lambda=1.7):
    """
    Cat Water Phobia Optimizer (CWPO) - Pseudocode Structure
    objf: Objective function
    lb: Lower bound
    ub: Upper bound
    dim: Number of dimensions
    SearchAgents_no: Population size
    Max_iter: Maximum iterations
    levy_lambda: Levy flight parameter
    """

    # Initialize population X with given bounds and random values
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = np.random.uniform(0, 1, SearchAgents_no) * (ub - lb) + lb

    # Initialize the best solution
    Alpha_pos = np.zeros(dim)
    Alpha_score = float("inf")

    Convergence_curve = np.zeros(Max_iter)
    s = solution()

    # Start optimization
    print('CWPO is optimizing "' + objf.__name__ + '"')
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(Max_iter):
        for i in range(SearchAgents_no):
            # Update the Levy Flight distribution
            levy_step = levy_flight(levy_lambda)

            # Compute Equation (1): Fitness value
            fitness = objf(Positions[i])

            # Compare with the best solution
            if Alpha_score < fitness:
                # Use Equation (1) to update the position
                Positions[i] = Alpha_pos + np.random.rand()
            else:
                # Use Equation (2) to update the position
                Positions[i] = Alpha_pos * levy_step + np.random.rand()

            # Ensure position stays within bounds
            Positions[i] = np.clip(Positions[i], lb, ub)

            # Update the best solution
            if fitness < Alpha_score:
                Alpha_score = fitness
                Alpha_pos = Positions[i].copy()

        Convergence_curve[l] = Alpha_score

        if (l + 1) % 500 == 0:
            print(f"At iteration {l + 1}, the best fitness is {Alpha_score}")

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "CWPO"
    s.bestIndividual = Alpha_pos
    s.objfname = objf.__name__

    return s
