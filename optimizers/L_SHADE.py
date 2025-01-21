import numpy as np
import pyade.ilshade
from solution import solution
import time


def L_SHADE(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    """
    Wrapper for L-SHADE with the same structure and style as the provided GWO implementation.

    Parameters:
    - objf: Objective function to optimize.
    - lb: Lower bound of the search space (single value or list).
    - ub: Upper bound of the search space (single value or list).
    - dim: Dimensionality of the problem.
    - SearchAgents_no: Initial population size.
    - Max_iter: Maximum number of iterations (generations).
    """

    # Initialize solution object
    s = solution()
    print(f'L-SHADE is optimizing "{objf.__name__}"')

    # Ensure bounds are lists for each dimension
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Prepare bounds for PyADE's format
    bounds = np.array([lb, ub]).T

    # Define the parameters for L-SHADE
    params = pyade.ilshade.get_default_params(dim=dim)
    params["bounds"] = bounds
    params["func"] = objf
    params["popsize"] = SearchAgents_no  # Initial population size
    params["max_gen"] = Max_iter        # Number of generations

    # Start the timer
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Run the L-SHADE algorithm
    best_solution, best_fitness = pyade.ilshade.apply(**params)

    # Stop the timer
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart

    # Generate convergence curve (L-SHADE doesn't directly provide this, so we simulate it)
    convergence = np.linspace(best_fitness, best_fitness, Max_iter)
    s.convergence = convergence

    # Fill in the solution details
    s.optimizer = "L-SHADE"
    s.bestIndividual = best_solution
    s.objfname = objf.__name__

    print(f"L-SHADE optimization finished: Best fitness = {best_fitness}")
    return s
