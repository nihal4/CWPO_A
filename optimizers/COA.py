import numpy as np
import time
from solution import solution

def COA(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    """
    Iguana-inspired metaheuristic optimization algorithm

    Parameters:
    - objf: Objective function to optimize.
    - lb: Lower bound of the search space (single value or list).
    - ub: Upper bound of the search space (single value or list).
    - dim: Dimensionality of the problem.
    - SearchAgents_no: Number of search agents (population size).
    - Max_iter: Maximum number of iterations (generations).

    Returns:
    - s: A solution object containing the results of the optimization process.
    """

    # Initialize solution object
    s = solution()
    print(f'COA is optimizing "{objf.__name__}"')

    """
    # Ensure bounds are lists for each dimension
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    lb = np.array(lb)
    ub = np.array(ub)
    """
    ub = np.array(ub)
    lb = np.array(lb)

    """
    if ub.shape == () and lb.shape == ():  # Scalars
        Positions = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb
    elif ub.shape[0] == dim and lb.shape[0] == dim:  # Arrays of length `dim`
        Positions = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb
    else:
        raise ValueError("Bounds `ub` and `lb` must be either scalars or arrays of the same length as `dim")
    """
    # Initialize the population
    X = np.random.uniform(lb, ub, (SearchAgents_no, dim))
    fit = np.array([objf(ind) for ind in X])

    # Initialize the best solution
    best_idx = np.argmin(fit)
    Xbest = X[best_idx].copy()
    fbest = fit[best_idx]

    # Start the timer
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Initialize convergence curve
    convergence_curve = np.zeros(Max_iter)

    # Main optimization loop
    for t in range(1, Max_iter + 1):
        for i in range(SearchAgents_no // 2):
            # Phase 1: Hunting and attacking strategy (Exploration phase)
            iguana = Xbest.copy()
            I = np.random.randint(1, 2)

            X_P1 = X[i] + np.random.rand() * (iguana - I * X[i])
            X_P1 = np.clip(X_P1, lb, ub)

            F_P1 = objf(X_P1)
            if F_P1 < fit[i]:
                X[i] = X_P1
                fit[i] = F_P1

        for i in range(SearchAgents_no // 2, SearchAgents_no):
            # Phase 1: Hunting and attacking strategy continued (Exploration phase)
            iguana = lb + np.random.rand(dim) * (ub - lb)
            F_HL = objf(iguana)
            I = np.random.randint(1, 2)

            if fit[i] > F_HL:
                X_P1 = X[i] + np.random.rand() * (iguana - I * X[i])
            else:
                X_P1 = X[i] + np.random.rand() * (X[i] - iguana)

            X_P1 = np.clip(X_P1, lb, ub)

            F_P1 = objf(X_P1)
            if F_P1 < fit[i]:
                X[i] = X_P1
                fit[i] = F_P1

        for i in range(SearchAgents_no):
            # Phase 2: Escaping from predators (Exploitation phase)
            LO_LOCAL = lb / t
            HI_LOCAL = ub / t

            X_P2 = X[i] + (1 - 2 * np.random.rand()) * (LO_LOCAL + np.random.rand() * (HI_LOCAL - LO_LOCAL))
            X_P2 = np.clip(X_P2, LO_LOCAL, HI_LOCAL)

            F_P2 = objf(X_P2)
            if F_P2 < fit[i]:
                X[i] = X_P2
                fit[i] = F_P2

        # Update the best solution
        best_idx = np.argmin(fit)
        if fit[best_idx] < fbest:
            fbest = fit[best_idx]
            Xbest = X[best_idx].copy()

        # Record convergence

        if (t+1) % 500 == 0:
            print(f'At iteration {t+1} the best solution fitness is {fbest}')
            
        convergence_curve[t - 1] = fbest

    # Stop the timer
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart

    # Fill in the solution details
    s.optimizer = "COA"
    s.objfname = objf.__name__
    s.convergence = convergence_curve
    s.best = fbest
    s.bestIndividual = Xbest

    return s
