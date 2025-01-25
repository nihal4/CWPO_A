import numpy as np
import random
import time
from solution import solution

def AO(F_obj, LB, UB, Dim, N, T):

    # Ensure UB and LB are numpy arrays
    UB = np.array(UB)
    LB = np.array(LB)

    Best_P = np.zeros(Dim)  # Ensure Best_P is a numpy array
    Best_FF = float("inf")

    # Initialize positions of search agents
    X = initialization(N, Dim, UB, LB)
    Xnew = X.copy()
    Ffun = np.zeros(N)
    Ffun_new = np.zeros(N)

    t = 1
    alpha = 0.1
    delta = 0.1

    # Record the convergence curve
    Convergence_curve = np.zeros(T)
    s = solution()

    # Start timer
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    print('AO is optimizing "' + F_obj.__name__ + '"')

    # Main loop
    while t <= T:
        for i in range(N):
            # Bound checking
            F_UB = X[i, :] > UB
            F_LB = X[i, :] < LB
            X[i, :] = (X[i, :] * ~(F_UB + F_LB)) + UB * F_UB + LB * F_LB

            # Calculate objective function for each agent
            Ffun[i] = F_obj(X[i, :])

            if Ffun[i] < Best_FF:
                Best_FF = Ffun[i]
                Best_P = X[i, :].copy()

        # Update the parameters G1 and G2
        G2 = 2 * random.random() - 1  # Eq. (16)
        G1 = 2 * (1 - (t / T))  # Eq. (17)
        to = np.arange(Dim)
        u = 0.0265
        r0 = 10
        r = r0 + u * to
        omega = 0.005
        phi0 = 3 * np.pi / 2
        phi = -omega * to + phi0
        x = r * np.sin(phi)  # Eq. (9)
        y = r * np.cos(phi)  # Eq. (10)
        QF = t ** ((2 * random.random() - 1) / (1 - T) ** 2)  # Eq. (15)

        # Main agent update loop
        for i in range(N):
            if t <= (2 / 3) * T:
                if random.random() < 0.5:
                    Xnew[i, :] = Best_P * (1 - t / T) + (np.mean(X[i, :]) - Best_P) * random.random()  # Eq. (3) and Eq. (4)
                    Ffun_new[i] = F_obj(Xnew[i, :])
                    if Ffun_new[i] < Ffun[i]:
                        X[i, :] = Xnew[i, :].copy()
                        Ffun[i] = Ffun_new[i]
                else:
                    Xnew[i, :] = Best_P * Levy(Dim) + X[int(np.floor(N * random.random()))] + (y - x) * random.random()  # Eq. (5)
                    Ffun_new[i] = F_obj(Xnew[i, :])
                    if Ffun_new[i] < Ffun[i]:
                        X[i, :] = Xnew[i, :].copy()
                        Ffun[i] = Ffun_new[i]
            else:
                if random.random() < 0.5:
                    Xnew[i, :] = (Best_P - np.mean(X, axis=0)) * alpha - random.random() + (UB - LB) * random.random() + LB * delta  # Eq. (13)
                    Ffun_new[i] = F_obj(Xnew[i, :])
                    if Ffun_new[i] < Ffun[i]:
                        X[i, :] = Xnew[i, :].copy()
                        Ffun[i] = Ffun_new[i]
                else:
                    Xnew[i, :] = QF * Best_P - G2 * X[i, :] * random.random() - G1 * Levy(Dim) + random.random() * G2  # Eq. (14)
                    Ffun_new[i] = F_obj(Xnew[i, :])
                    if Ffun_new[i] < Ffun[i]:
                        X[i, :] = Xnew[i, :].copy()
                        Ffun[i] = Ffun_new[i]

        # Convergence update
        if t % 500 == 0:
            print(f'At iteration {t} the best solution fitness is {Best_FF}')

        Convergence_curve[t - 1] = Best_FF
        t += 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "AO"
    s.bestIndividual = Best_P
    s.objfname = F_obj.__name__

    return s

def Levy(d):
    beta = 1.5
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(d) * sigma
    v = np.random.randn(d)
    step = u / np.abs(v) ** (1 / beta)
    return step

def initialization(N, Dim, UB, LB):
    """Random initialization of search agents' positions."""
    return np.random.uniform(LB, UB, (N, Dim))

'''
#for F1-Gt1
import numpy as np
import random
import time
#from scipy.special import gamma
from solution import solution

def AO(F_obj, LB, UB, Dim, N, T):

    Best_P = np.zeros(Dim)
    Best_FF = float("inf")

    # Initialize positions of search agents
    X = initialization(N, Dim, UB, LB)
    Xnew = X
    Ffun = np.zeros(N)
    Ffun_new = np.zeros(N)

    t = 1
    alpha = 0.1
    delta = 0.1

    # Record the convergence curve
    Convergence_curve = np.zeros(T)
    s = solution()

    # Start timer
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    print('AO is optimizing "' + F_obj.__name__ + '"')

    # Main loop
    while t <= T:
        for i in range(N):
            # Bound checking
            F_UB = X[i, :] > UB
            F_LB = X[i, :] < LB
            X[i, :] = (X[i, :] * ~(F_UB + F_LB)) + UB * F_UB + LB * F_LB

            # Calculate objective function for each agent
            Ffun[i] = F_obj(X[i, :])

            if Ffun[i] < Best_FF:
                Best_FF = Ffun[i]
                Best_P = X[i, :]

        # Update the parameters G1 and G2
        G2 = 2 * random.random() - 1  # Eq. (16)
        G1 = 2 * (1 - (t / T))  # Eq. (17)
        to = np.arange(Dim)
        u = 0.0265
        r0 = 10
        r = r0 + u * to
        omega = 0.005
        phi0 = 3 * np.pi / 2
        phi = -omega * to + phi0
        x = r * np.sin(phi)  # Eq. (9)
        y = r * np.cos(phi)  # Eq. (10)
        QF = t ** ((2 * random.random() - 1) / (1 - T) ** 2)  # Eq. (15)

        # Main agent update loop
        for i in range(N):
            if t <= (2 / 3) * T:
                if random.random() < 0.5:
                    Xnew[i, :] = Best_P * (1 - t / T) + (np.mean(X[i, :]) - Best_P) * random.random()  # Eq. (3) and Eq. (4)
                    Ffun_new[i] = F_obj(Xnew[i, :])
                    if Ffun_new[i] < Ffun[i]:
                        X[i, :] = Xnew[i, :]
                        Ffun[i] = Ffun_new[i]
                else:
                    Xnew[i, :] = Best_P * Levy(Dim) + X[int(np.floor(N * random.random()))] + (y - x) * random.random()  # Eq. (5)
                    Ffun_new[i] = F_obj(Xnew[i, :])
                    if Ffun_new[i] < Ffun[i]:
                        X[i, :] = Xnew[i, :]
                        Ffun[i] = Ffun_new[i]
            else:
                if random.random() < 0.5:
                    Xnew[i, :] = (Best_P - np.mean(X, axis=0)) * alpha - random.random() + (UB - LB) * random.random() + LB * delta  # Eq. (13)
                    Ffun_new[i] = F_obj(Xnew[i, :])
                    if Ffun_new[i] < Ffun[i]:
                        X[i, :] = Xnew[i, :]
                        Ffun[i] = Ffun_new[i]
                else:
                    Xnew[i, :] = QF * Best_P - G2 * X[i, :] * random.random() - G1 * Levy(Dim) + random.random() * G2  # Eq. (14)
                    Ffun_new[i] = F_obj(Xnew[i, :])
                    if Ffun_new[i] < Ffun[i]:
                        X[i, :] = Xnew[i, :]
                        Ffun[i] = Ffun_new[i]

        # Convergence update
        if t % 500 == 0:
            print(f'At iteration {t} the best solution fitness is {Best_FF}')

        Convergence_curve[t - 1] = Best_FF
        t += 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "AO"
    s.bestIndividual = Best_P
    s.objfname = F_obj.__name__

    return s


def Levy(d):
    beta = 1.5
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(d) * sigma
    v = np.random.randn(d)
    step = u / np.abs(v) ** (1 / beta)
    return step
'''

def initialization(N, Dim, UB, LB):
    """Random initialization of search agents' positions."""
    return np.random.uniform(LB, UB, (N, Dim))
