
"""
#V4
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
    #Rotate the position vector by the given angle (in degrees).#
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
    
    #Cat Water Phobia Optimizer (CWPO) with 180-degree position search.
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
            # Dynamic Levy Flight step based on fitness
            levy_step = levy_flight(levy_lambda)

            # Calculate dynamic scaling for Levy step
            if Alpha_score > 0:  # Prevent division by zero
                levy_step = levy_step * (Alpha_score / (objf(Positions[i]) + 1e-6))  # Scaling based on fitness

            # Ensure the step does not exceed the best fitness
            levy_step = min(levy_step, Alpha_score)

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
