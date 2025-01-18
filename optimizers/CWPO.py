import numpy as np
import time
from solution import solution


# Initialize the population of stray cats randomly
def initialize_population(num_cats, bounds):
    lower_bounds, upper_bounds = bounds[0], bounds[1]
    return np.random.uniform(lower_bounds, upper_bounds, (num_cats, len(lower_bounds)))


# Simulate environmental hazards
def calculate_hazards(cat_pos, time, alpha, beta, omega):
    # Restrict cat positions to the first 2 dimensions
    cat_pos_2d = cat_pos[:, :2]
    
    # Define hazard source in 2D
    hazard_source = np.array([5 * np.sin(omega * time), 5 * np.cos(omega * time)])
    
    # Calculate distances
    distances = np.linalg.norm(cat_pos_2d - hazard_source, axis=1)
    
    # Compute hazards
    return alpha * (distances ** -1) + beta * np.cos(omega * time)



# Update the position of stray cats using Levy flight
def update_positions(cat_pos, hazards, levy_lambda, bounds):
    levy_flights = np.random.standard_cauchy(cat_pos.shape) * levy_lambda
    new_positions = cat_pos + levy_flights - hazards[:, None]
    new_positions = np.clip(new_positions, bounds[0], bounds[1])  # Keep positions within bounds
    return new_positions


# Local resource search
def localized_search(cat_pos, resources, sf):
    distances = np.linalg.norm(resources - cat_pos[:, None], axis=2)
    influences = np.exp(-distances / sf)  # Exponential decay for influence
    return np.sum(influences, axis=1)


# Optimization algorithm
def CWPO(objf, lb, ub, dim, SearchAgents_no, Max_iter, alpha=1, beta=0.5, omega_freq=2 * np.pi / 10, sf=2, levy_lambda=1.5):
    """
    Cat Water Phobia Optimizer (CWPO)
    objf: Objective function
    lb: Lower bound
    ub: Upper bound
    dim: Number of dimensions
    SearchAgents_no: Population size
    Max_iter: Maximum iterations
    alpha: Hazard scaling factor
    beta: Hazard fluctuation factor
    omega_freq: Frequency of hazard oscillation (f), resulting in omega = 2 * pi * f
    sf: Search radius
    levy_lambda: Levy flight parameter
    """

    # Calculate angular frequency
    omega = 2 * np.pi * omega_freq
    bounds = ([lb] * dim, [ub] * dim)

    # Initialize population and resources
    cat_positions = initialize_population(SearchAgents_no, bounds)
    resources = np.random.uniform(bounds[0], bounds[1], (SearchAgents_no, dim))

    # Initialize the best solution
    Alpha_pos = np.zeros(dim)
    Alpha_score = float("inf")
    Convergence_curve = np.zeros(Max_iter)
    s = solution()

    # Start optimization
    print('CWPO is optimizing "' + objf.__name__ + '"')
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for t in range(Max_iter):
        # Calculate hazards for the current iteration
        hazards = calculate_hazards(cat_positions, t, alpha, beta, omega)

        # Update positions using Levy flight and hazard effects
        cat_positions = update_positions(cat_positions, hazards, levy_lambda, bounds)

        # Perform localized search
        local_influences = localized_search(cat_positions, resources, sf)

        # Calculate fitness
        scores = np.apply_along_axis(objf, 1, cat_positions)
        current_best_score = np.min(scores)
        global_best_index = np.argmin(scores)

        # Update the best solution
        if current_best_score < Alpha_score:
            Alpha_score = current_best_score
            Alpha_pos = cat_positions[global_best_index].copy()

        # Store the best score for convergence analysis
        Convergence_curve[t] = Alpha_score

        if t % 1 == 0:
            print(f"At iteration {t + 1}, the best fitness is {Alpha_score}")

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "CWPO"
    s.bestIndividual = Alpha_pos
    s.objfname = objf.__name__

    return s
