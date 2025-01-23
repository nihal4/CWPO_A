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

def rotate_position(position, angle):
    """Rotate the position vector by the given angle (in degrees)."""
    radians = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians), np.cos(radians)]
    ])
    if len(position) >= 2:  # Ensure rotation only applies to the first two dimensions
        rotated = np.dot(rotation_matrix, position[:2])
        return np.concatenate((rotated, position[2:]))
    return position

def CWPO(objf, lb, ub, dim, SearchAgents_no, Max_iter, levy_lambda_min=0.5, levy_lambda_max=1.7):
    """
    Cat Water Phobia Optimizer (CWPO) with dynamic Levy flight parameter and revisit prevention.
    
    objf: Objective function
    lb: Lower bound
    ub: Upper bound
    dim: Number of dimensions
    SearchAgents_no: Population size
    Max_iter: Maximum iterations
    levy_lambda_min: Minimum Levy flight parameter
    levy_lambda_max: Maximum Levy flight parameter
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

    # Track visited positions
    visited_positions = set()

    # Start optimization
    print('CWPO is optimizing "' + objf.__name__ + '"')
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(Max_iter):
        for i in range(SearchAgents_no):
            # Save the current position as the best so far
            current_position = Positions[i].copy()
            current_fitness = objf(current_position)

            # Try Levy flight to explore new position
            levy_lambda = np.random.uniform(levy_lambda_min, levy_lambda_max)  # Varying Levy flight parameter
            while True:
                levy_step = levy_flight(levy_lambda)
                new_position = Alpha_pos * levy_step + np.random.rand(dim)
                new_position = np.clip(new_position, lb, ub)  # Ensure bounds

                # Prevent revisiting the same position
                new_position_tuple = tuple(np.round(new_position, decimals=5))  # Using rounded tuple for precision
                if new_position_tuple in visited_positions:
                    continue  # Skip if the position has been visited before

                # Evaluate fitness of the new position
                new_fitness = objf(new_position)

                if new_fitness < current_fitness:
                    # If new position improves fitness, accept it
                    Positions[i] = new_position
                    visited_positions.add(new_position_tuple)
                    break  # Exit the retry loop
                else:
                    # If no improvement, try perturbing the position slightly
                    new_position = current_position + np.random.uniform(-0.01, 0.01, dim)
                    new_position = np.clip(new_position, lb, ub)

            # Perform 180-degree search for better local position
            best_fitness_in_range = objf(Positions[i])
            best_position_in_range = Positions[i].copy()

            for angle in range(0, 361):  # 0 to 360 degrees (inclusive)
                candidate_position = rotate_position(Positions[i], angle)
                candidate_position = np.clip(candidate_position, lb, ub)  # Keep within bounds
                candidate_fitness = objf(candidate_position)

                if candidate_fitness < best_fitness_in_range:
                    best_fitness_in_range = candidate_fitness
                    best_position_in_range = candidate_position

            # Update global position if better fitness is found
            if best_fitness_in_range < Alpha_score:
                Alpha_score = best_fitness_in_range
                Alpha_pos = best_position_in_range.copy()

        Convergence_curve[l] = Alpha_score

        if (l + 1) % 1 == 0:
            print(f"At iteration {l + 1}, the best fitness is {Alpha_score}")

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "CWPO"
    s.bestIndividual = Alpha_pos
    s.objfname = objf.__name__

    return s


"""
#v3
import numpy as np
import time
from solution import solution


def levy_flight(lam):
    #Generate a Levy flight step.
    sigma = (np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2) /
             (np.math.gamma((1 + lam) / 2) * lam * 2 ** ((lam - 1) / 2))) ** (1 / lam)
    u = np.random.normal(0, sigma, 1)
    v = np.random.normal(0, 1, 1)
    step = u / abs(v) ** (1 / lam)
    return step[0]


def rotate_position(position, angle):
    #Rotate the position vector by the given angle (in degrees).
    radians = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians), np.cos(radians)]
    ])
    if len(position) >= 2:  # Ensure rotation only applies to the first two dimensions
        rotated = np.dot(rotation_matrix, position[:2])
        return np.concatenate((rotated, position[2:]))
    return position


def CWPO(objf, lb, ub, dim, SearchAgents_no, Max_iter, levy_lambda=1.7):
    
    #Cat Water Phobia Optimizer (CWPO) with 180-degree position search
    #objf: Objective function
    #lb: Lower bound
    #ub: Upper bound
    #dim: Number of dimensions
    #SearchAgents_no: Population size
    #Max_iter: Maximum iterations
    #levy_lambda: Levy flight parameter
    

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

            # Perform 180-degree search
            best_fitness_in_range = objf(Positions[i])
            best_position_in_range = Positions[i].copy()

            for angle in range(0, 361):  # 0 to 180 degrees
                candidate_position = rotate_position(Positions[i], angle)
                candidate_position = np.clip(candidate_position, lb, ub)  # Keep within bounds
                candidate_fitness = objf(candidate_position)

                if candidate_fitness < best_fitness_in_range:
                    best_fitness_in_range = candidate_fitness
                    best_position_in_range = candidate_position

            # Update position based on the fitness comparison
            if best_fitness_in_range < Alpha_score:
                # Use Equation (1) to update the position
                Positions[i] = Alpha_pos + np.random.rand()
            else:
                # Use Equation (2) to update the position
                Positions[i] = Alpha_pos * levy_step + np.random.rand()

            # Ensure position stays within bounds
            Positions[i] = np.clip(Positions[i], lb, ub)

            # Update the best solution
            if best_fitness_in_range < Alpha_score:
                Alpha_score = best_fitness_in_range
                Alpha_pos = best_position_in_range.copy()

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

"""
"""
#version V2
import numpy as np
import time
from solution import solution


def levy_flight(lam):
    #Generate a Levy flight step.
    sigma = (np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2) /
             (np.math.gamma((1 + lam) / 2) * lam * 2 ** ((lam - 1) / 2))) ** (1 / lam)
    u = np.random.normal(0, sigma, 1)
    v = np.random.normal(0, 1, 1)
    step = u / abs(v) ** (1 / lam)
    return step[0]


def CWPO(objf, lb, ub, dim, SearchAgents_no, Max_iter, levy_lambda=1.7):
    
    #Cat Water Phobia Optimizer (CWPO) - Pseudocode Structure
    #objf: Objective function
    #lb: Lower bound
    #ub: Upper bound
    #dim: Number of dimensions
    #SearchAgents_no: Population size
    #Max_iter: Maximum iterations
    #levy_lambda: Levy flight parameter
    

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
"""
