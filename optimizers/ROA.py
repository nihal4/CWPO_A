import numpy as np
import random
import time
from solution import solution

def init(Search_Agents, dimensions, Upperbound, Lowerbound):
    return np.random.uniform(Lowerbound, Upperbound, (Search_Agents, dimensions))

def ROA(objf, lb, ub, dim, SearchAgents_no, Max_iter):

    BestRemora = np.zeros(dim)
    Score = float("inf")

    # Initialize remora population
    Remora = init(SearchAgents_no, dim, ub, lb)
    Prevgen = [Remora]
    Convergence_curve = np.zeros(Max_iter)
    s = solution()

    print(f'ROA is optimizing "{objf.__name__}"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    t = 0
    while t < Max_iter:

        # Memory of previous generation
        PreviousRemora = Prevgen[t - 1] if t > 0 else Prevgen[0]

        # Boundary check and fitness evaluation
        for i in range(Remora.shape[0]):
            Remora[i, :] = np.clip(Remora[i, :], lb, ub)
            fitness = objf(Remora[i, :])

            if fitness < Score:
                Score = fitness
                BestRemora = Remora[i, :].copy()

        # Make an experience attempt (Equation 2)
        for j in range(Remora.shape[0]):
            RemoraAtt = Remora[j, :] + (Remora[j, :] - PreviousRemora[j, :]) * np.random.randn()

            # Fitness evaluation of attempted solution
            fitnessAtt = objf(RemoraAtt)
            fitnessI = objf(Remora[j, :])

            if fitnessI > fitnessAtt:
                V = 2 * (1 - t / Max_iter)  # Equation (12)
                B = 2 * V * random.random() - V  # Equation (11)
                C = 0.1
                A = B * (Remora[j, :] - C * BestRemora)  # Equation (10)
                Remora[j, :] = Remora[j, :] + A  # Equation (9)
            elif random.randint(0, 1) == 0:
                a = -(1 + t / Max_iter)  # Equation (7)
                alpha = random.random() * (a - 1) + 1  # Equation (6)
                D = np.abs(BestRemora - Remora[j, :])  # Equation (8)
                Remora[j, :] = D * np.exp(alpha) * np.cos(2 * np.pi * a) + Remora[j, :]  # Equation (5)
            else:
                m = random.sample(range(Remora.shape[0]), 1)[0]
                Remora[j, :] = BestRemora - (
                    (random.random() * (BestRemora + Remora[m, :]) / 2) - Remora[m, :]
                )  # Equation (1)
        if (t+1) % 500 == 0:
            print(f"Iteration : {t+1}, best score : {Score}")
            
        t += 1
        
        Prevgen.append(Remora.copy())
        Convergence_curve[t - 1] = Score

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "ROA"
    s.bestIndividual = BestRemora
    s.objfname = objf.__name__

    return s
