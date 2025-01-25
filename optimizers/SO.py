
import numpy as np
import random
import time
from solution import solution

def SO(fobj, lb, ub, dim, N, T):
    # Initialize population
    X = np.random.uniform(lb, ub, (N, dim))
    fitness = np.zeros(N)

    # Evaluate fitness of each agent
    for i in range(N):
        fitness[i] = fobj(X[i, :])

    # Best food initialization
    GYbest = np.min(fitness)
    gbest = np.argmin(fitness)
    Xfood = X[gbest, :]

    # Dividing the swarm into two equal groups: males and females
    Nm = round(N / 2)  # Eq. (2 & 3)
    Nf = N - Nm
    Xm = X[:Nm, :]
    Xf = X[Nm:, :]
    fitness_m = fitness[:Nm]
    fitness_f = fitness[Nm:]

    # Best male and female positions
    fitnessBest_m, gbest1 = np.min(fitness_m), np.argmin(fitness_m)
    Xbest_m = Xm[gbest1, :]
    fitnessBest_f, gbest2 = np.min(fitness_f), np.argmin(fitness_f)
    Xbest_f = Xf[gbest2, :]

    # Main loop
    vec_flag = [1, -1]
    Threshold = 0.25
    Thresold2 = 0.6
    C1 = 0.5
    C2 = 0.05
    C3 = 2

    s = solution()
    print(f'Swarm Optimization is optimizing "{fobj.__name__}"')
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    gbest_t = np.zeros(T)
    
    # Main optimization loop
    for t in range(T):
        Temp = np.exp(-(t) / T)  # Eq. (4)
        Q = C1 * np.exp(((t - T) / T))  # Eq. (5)
        Q = min(Q, 1)

        # Exploration Phase (no food)
        if Q < Threshold:
            for i in range(Nm):
                for j in range(dim):
                    rand_leader_index = random.randint(0, Nm-1)
                    X_randm = Xm[rand_leader_index, :]
                    flag_index = random.randint(0, 1)
                    Flag = vec_flag[flag_index]
                    Am = np.exp(-fitness_m[rand_leader_index] / (fitness_m[i] + np.finfo(float).eps))  # Eq. (7)
                    Xm[i, j] = X_randm[j] + Flag * C2 * Am * ((ub[j] - lb[j]) * np.random.rand() + lb[j])  # Eq. (6)

            for i in range(Nf):
                for j in range(dim):
                    rand_leader_index = random.randint(0, Nf-1)
                    X_randf = Xf[rand_leader_index, :]
                    flag_index = random.randint(0, 1)
                    Flag = vec_flag[flag_index]
                    Af = np.exp(-fitness_f[rand_leader_index] / (fitness_f[i] + np.finfo(float).eps))  # Eq. (9)
                    Xf[i, j] = X_randf[j] + Flag * C2 * Af * ((ub[j] - lb[j]) * np.random.rand() + lb[j])  # Eq. (8)
        else:  # Exploitation Phase (food exists)
            if Temp > Thresold2:  # hot
                for i in range(Nm):
                    flag_index = random.randint(0, 1)
                    Flag = vec_flag[flag_index]
                    for j in range(dim):
                        Xm[i, j] = Xfood[j] + C3 * Flag * Temp * np.random.rand() * (Xfood[j] - Xm[i, j])  # Eq. (10)

                for i in range(Nf):
                    flag_index = random.randint(0, 1)
                    Flag = vec_flag[flag_index]
                    for j in range(dim):
                        Xf[i, j] = Xfood[j] + Flag * C3 * Temp * np.random.rand() * (Xfood[j] - Xf[i, j])  # Eq. (10)
            else:  # cold
                if np.random.rand() > 0.6:  # fight
                    for i in range(Nm):
                        for j in range(dim):
                            FM = np.exp(-fitnessBest_f / (fitness_m[i] + np.finfo(float).eps))  # Eq. (13)
                            Xm[i, j] = Xm[i, j] + C3 * FM * np.random.rand() * (Q * Xbest_f[j] - Xm[i, j])  # Eq. (11)

                    for i in range(Nf):
                        for j in range(dim):
                            FF = np.exp(-fitnessBest_m / (fitness_f[i] + np.finfo(float).eps))  # Eq. (14)
                            Xf[i, j] = Xf[i, j] + C3 * FF * np.random.rand() * (Q * Xbest_m[j] - Xf[i, j])  # Eq. (12)
                else:  # mating
                    for i in range(Nm):
                        for j in range(dim):
                            Mm = np.exp(-fitness_f[i] / (fitness_m[i] + np.finfo(float).eps))  # Eq. (17)
                            Xm[i, j] = Xm[i, j] + C3 * np.random.rand() * Mm * (Q * Xf[i, j] - Xm[i, j])  # Eq. (15)

                    for i in range(Nf):
                        for j in range(dim):
                            Mf = np.exp(-fitness_m[i] / (fitness_f[i] + np.finfo(float).eps))  # Eq. (18)
                            Xf[i, j] = Xf[i, j] + C3 * np.random.rand() * Mf * (Q * Xm[i, j] - Xf[i, j])  # Eq. (16)

                    flag_index = random.randint(0, 1)
                    egg = vec_flag[flag_index]
                    if egg == 1:
                        worst_m = np.argmax(fitness_m)
                        Xm[worst_m, :] = lb + np.random.rand() * (ub - lb)  # Eq. (19)

                        worst_f = np.argmax(fitness_f)
                        Xf[worst_f, :] = lb + np.random.rand() * (ub - lb)  # Eq. (20)

        # Update the positions and evaluate the fitness
        for j in range(Nm):
            Xnewm = np.clip(Xm[j, :], lb, ub)
            y = fobj(Xnewm)
            if y < fitness_m[j]:
                fitness_m[j] = y
                Xm[j, :] = Xnewm

        fitnessBest_m, gbest1 = np.min(fitness_m), np.argmin(fitness_m)

        for j in range(Nf):
            Xnewf = np.clip(Xf[j, :], lb, ub)
            y = fobj(Xnewf)
            if y < fitness_f[j]:
                fitness_f[j] = y
                Xf[j, :] = Xnewf

        fitnessBest_f, gbest2 = np.min(fitness_f), np.argmin(fitness_f)

        # Update the best positions
        if fitnessBest_m < fitnessBest_f:
            GYbest = fitnessBest_m
            Xfood = Xbest_m
        else:
            GYbest = fitnessBest_f
            Xfood = Xbest_f

        gbest_t[t] = GYbest
        if (t+1) % 500 == 0:
            print(f"Iteration {t+1}: Best fitness = {GYbest}")

    # Timer end
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = gbest_t
    s.optimizer = "SO"
    s.bestIndividual = Xfood
    s.objfname = fobj.__name__

    return s
"""

#for F1-23
import numpy as np
import random
import time
from solution import solution

def SO(fobj, lb, ub, dim, N, T):
    # Initialize population
    X = np.random.uniform(lb, ub, (N, dim))
    fitness = np.zeros(N)

    # Evaluate fitness of each agent
    for i in range(N):
        fitness[i] = fobj(X[i, :])

    # Best food initialization
    GYbest = np.min(fitness)
    gbest = np.argmin(fitness)
    Xfood = X[gbest, :]

    # Dividing the swarm into two equal groups: males and females
    Nm = round(N / 2)  # Eq. (2 & 3)
    Nf = N - Nm
    Xm = X[:Nm, :]
    Xf = X[Nm:, :]
    fitness_m = fitness[:Nm]
    fitness_f = fitness[Nm:]

    # Best male and female positions
    fitnessBest_m, gbest1 = np.min(fitness_m), np.argmin(fitness_m)
    Xbest_m = Xm[gbest1, :]
    fitnessBest_f, gbest2 = np.min(fitness_f), np.argmin(fitness_f)
    Xbest_f = Xf[gbest2, :]

    # Main loop
    vec_flag = [1, -1]
    Threshold = 0.25
    Thresold2 = 0.6
    C1 = 0.5
    C2 = 0.05
    C3 = 2

    s = solution()
    print(f'Swarm Optimization is optimizing "{fobj.__name__}"')
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    gbest_t = np.zeros(T)
    
    # Main optimization loop
    for t in range(T):
        Temp = np.exp(-(t) / T)  # Eq. (4)
        Q = C1 * np.exp(((t - T) / T))  # Eq. (5)
        Q = min(Q, 1)

        # Exploration Phase (no food)
        if Q < Threshold:
            for i in range(Nm):
                for j in range(dim):
                    rand_leader_index = random.randint(0, Nm-1)
                    X_randm = Xm[rand_leader_index, :]
                    flag_index = random.randint(0, 1)
                    Flag = vec_flag[flag_index]
                    Am = np.exp(-fitness_m[rand_leader_index] / (fitness_m[i] + np.finfo(float).eps))  # Eq. (7)
                    Xm[i, j] = X_randm[j] + Flag * C2 * Am * ((ub - lb) * np.random.rand() + lb)  # Eq. (6)

            for i in range(Nf):
                for j in range(dim):
                    rand_leader_index = random.randint(0, Nf-1)
                    X_randf = Xf[rand_leader_index, :]
                    flag_index = random.randint(0, 1)
                    Flag = vec_flag[flag_index]
                    Af = np.exp(-fitness_f[rand_leader_index] / (fitness_f[i] + np.finfo(float).eps))  # Eq. (9)
                    Xf[i, j] = X_randf[j] + Flag * C2 * Af * ((ub - lb) * np.random.rand() + lb)  # Eq. (8)
        else:  # Exploitation Phase (food exists)
            if Temp > Thresold2:  # hot
                for i in range(Nm):
                    flag_index = random.randint(0, 1)
                    Flag = vec_flag[flag_index]
                    for j in range(dim):
                        Xm[i, j] = Xfood[j] + C3 * Flag * Temp * np.random.rand() * (Xfood[j] - Xm[i, j])  # Eq. (10)

                for i in range(Nf):
                    flag_index = random.randint(0, 1)
                    Flag = vec_flag[flag_index]
                    for j in range(dim):
                        Xf[i, j] = Xfood[j] + Flag * C3 * Temp * np.random.rand() * (Xfood[j] - Xf[i, j])  # Eq. (10)
            else:  # cold
                if np.random.rand() > 0.6:  # fight
                    for i in range(Nm):
                        for j in range(dim):
                            FM = np.exp(-fitnessBest_f / (fitness_m[i] + np.finfo(float).eps))  # Eq. (13)
                            Xm[i, j] = Xm[i, j] + C3 * FM * np.random.rand() * (Q * Xbest_f[j] - Xm[i, j])  # Eq. (11)

                    for i in range(Nf):
                        for j in range(dim):
                            FF = np.exp(-fitnessBest_m / (fitness_f[i] + np.finfo(float).eps))  # Eq. (14)
                            Xf[i, j] = Xf[i, j] + C3 * FF * np.random.rand() * (Q * Xbest_m[j] - Xf[i, j])  # Eq. (12)
                else:  # mating
                    for i in range(Nm):
                        for j in range(dim):
                            Mm = np.exp(-fitness_f[i] / (fitness_m[i] + np.finfo(float).eps))  # Eq. (17)
                            Xm[i, j] = Xm[i, j] + C3 * np.random.rand() * Mm * (Q * Xf[i, j] - Xm[i, j])  # Eq. (15)

                    for i in range(Nf):
                        for j in range(dim):
                            Mf = np.exp(-fitness_m[i] / (fitness_f[i] + np.finfo(float).eps))  # Eq. (18)
                            Xf[i, j] = Xf[i, j] + C3 * np.random.rand() * Mf * (Q * Xm[i, j] - Xf[i, j])  # Eq. (16)

                    flag_index = random.randint(0, 1)
                    egg = vec_flag[flag_index]
                    if egg == 1:
                        worst_m = np.argmax(fitness_m)
                        Xm[worst_m, :] = lb + np.random.rand() * (ub - lb)  # Eq. (19)

                        worst_f = np.argmax(fitness_f)
                        Xf[worst_f, :] = lb + np.random.rand() * (ub - lb)  # Eq. (20)

        # Update the positions and evaluate the fitness
        for j in range(Nm):
            Xnewm = np.clip(Xm[j, :], lb, ub)
            y = fobj(Xnewm)
            if y < fitness_m[j]:
                fitness_m[j] = y
                Xm[j, :] = Xnewm

        fitnessBest_m, gbest1 = np.min(fitness_m), np.argmin(fitness_m)

        for j in range(Nf):
            Xnewf = np.clip(Xf[j, :], lb, ub)
            y = fobj(Xnewf)
            if y < fitness_f[j]:
                fitness_f[j] = y
                Xf[j, :] = Xnewf

        fitnessBest_f, gbest2 = np.min(fitness_f), np.argmin(fitness_f)

        # Update the best positions
        if fitnessBest_m < fitnessBest_f:
            GYbest = fitnessBest_m
            Xfood = Xbest_m
        else:
            GYbest = fitnessBest_f
            Xfood = Xbest_f

        gbest_t[t] = GYbest
        if (t+1)%500==0:
            print(f"Iteration {t+1}: Best fitness = {GYbest}")

    # Timer end
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = gbest_t
    s.optimizer = "SO"
    s.bestIndividual = Xfood
    s.objfname = fobj.__name__

    return s
"""

