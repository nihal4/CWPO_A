# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:46:20 2016

@author: Hossam Faris
"""

import numpy
import math

# define the function blocks
def prod(it):
    p = 1
    for n in it:
        p *= n
    return p


def Ufun(x, a, k, m):
    y = k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))
    return y


def F1(x):
    s = numpy.sum(x ** 2)
    return s


def F2(x):
    o = sum(abs(x)) + prod(abs(x))
    return o


def F3(x):
    dim = len(x) + 1
    o = 0
    for i in range(1, dim):
        o = o + (numpy.sum(x[0:i])) ** 2
    return o


def F4(x):
    o = max(abs(x))
    return o


def F5(x):
    dim = len(x)
    o = numpy.sum(
        100 * (x[1:dim] - (x[0 : dim - 1] ** 2)) ** 2 + (x[0 : dim - 1] - 1) ** 2
    )
    return o


def F6(x):
    o = numpy.sum(abs((x + 0.5)) ** 2)
    return o


def F7(x):
    dim = len(x)

    w = [i for i in range(len(x))]
    for i in range(0, dim):
        w[i] = i + 1
    o = numpy.sum(w * (x ** 4)) + numpy.random.uniform(0, 1)
    return o


def F8(x):
    o = sum(-x * (numpy.sin(numpy.sqrt(abs(x)))))
    return o


def F9(x):
    dim = len(x)
    o = numpy.sum(x ** 2 - 10 * numpy.cos(2 * math.pi * x)) + 10 * dim
    return o


def F10(x):
    dim = len(x)
    o = (
        -20 * numpy.exp(-0.2 * numpy.sqrt(numpy.sum(x ** 2) / dim))
        - numpy.exp(numpy.sum(numpy.cos(2 * math.pi * x)) / dim)
        + 20
        + numpy.exp(1)
    )
    return o


def F11(x):
    dim = len(x)
    w = [i for i in range(len(x))]
    w = [i + 1 for i in w]
    o = numpy.sum(x ** 2) / 4000 - prod(numpy.cos(x / numpy.sqrt(w))) + 1
    return o


def F12(x):
    dim = len(x)
    o = (math.pi / dim) * (
        10 * ((numpy.sin(math.pi * (1 + (x[0] + 1) / 4))) ** 2)
        + numpy.sum(
            (((x[: dim - 1] + 1) / 4) ** 2)
            * (1 + 10 * ((numpy.sin(math.pi * (1 + (x[1 :] + 1) / 4)))) ** 2)
        )
        + ((x[dim - 1] + 1) / 4) ** 2
    ) + numpy.sum(Ufun(x, 10, 100, 4))
    return o


def F13(x):
    if x.ndim==1:
        x = x.reshape(1,-1)

    o = 0.1 * (
        (numpy.sin(3 * numpy.pi * x[:,0])) ** 2
        + numpy.sum(
            (x[:,:-1] - 1) ** 2
            * (1 + (numpy.sin(3 * numpy.pi * x[:,1:])) ** 2), axis=1
        )
        + ((x[:,-1] - 1) ** 2) * (1 + (numpy.sin(2 * numpy.pi * x[:,-1])) ** 2)
    ) + numpy.sum(Ufun(x, 5, 100, 4))
    return o


def F14(x):
    aS = [
        [
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
        ],
        [
            -32, -32, -32, -32, -32,
            -16, -16, -16, -16, -16,
            0, 0, 0, 0, 0,
            16, 16, 16, 16, 16,
            32, 32, 32, 32, 32,
        ],
    ]
    aS = numpy.asarray(aS)
    bS = numpy.zeros(25)
    v = numpy.matrix(x)
    for i in range(0, 25):
        H = v - aS[:, i]
        bS[i] = numpy.sum((numpy.power(H, 6)))
    w = [i for i in range(25)]
    for i in range(0, 24):
        w[i] = i + 1
    o = ((1.0 / 500) + numpy.sum(1.0 / (w + bS))) ** (-1)
    return o


def F15(L):
    aK = [
        0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627,
        0.0456, 0.0342, 0.0323, 0.0235, 0.0246,
    ]
    bK = [0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16]
    aK = numpy.asarray(aK)
    bK = numpy.asarray(bK)
    bK = 1 / bK
    fit = numpy.sum(
        (aK - ((L[0] * (bK ** 2 + L[1] * bK)) / (bK ** 2 + L[2] * bK + L[3]))) ** 2
    )
    return fit


def F16(L):
    o = (
        4 * (L[0] ** 2)
        - 2.1 * (L[0] ** 4)
        + (L[0] ** 6) / 3
        + L[0] * L[1]
        - 4 * (L[1] ** 2)
        + 4 * (L[1] ** 4)
    )
    return o


def F17(L):
    o = (
        (L[1] - (L[0] ** 2) * 5.1 / (4 * (numpy.pi ** 2)) + 5 / numpy.pi * L[0] - 6)
        ** 2
        + 10 * (1 - 1 / (8 * numpy.pi)) * numpy.cos(L[0])
        + 10
    )
    return o


def F18(L):
    o = (
        1
        + (L[0] + L[1] + 1) ** 2
        * (
            19
            - 14 * L[0]
            + 3 * (L[0] ** 2)
            - 14 * L[1]
            + 6 * L[0] * L[1]
            + 3 * L[1] ** 2
        )
    ) * (
        30
        + (2 * L[0] - 3 * L[1]) ** 2
        * (
            18
            - 32 * L[0]
            + 12 * (L[0] ** 2)
            + 48 * L[1]
            - 36 * L[0] * L[1]
            + 27 * (L[1] ** 2)
        )
    )
    return o


# map the inputs to the function blocks
def F19(L):
    aH = [[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]]
    aH = numpy.asarray(aH)
    cH = [1, 1.2, 3, 3.2]
    cH = numpy.asarray(cH)
    pH = [
        [0.3689, 0.117, 0.2673],
        [0.4699, 0.4387, 0.747],
        [0.1091, 0.8732, 0.5547],
        [0.03815, 0.5743, 0.8828],
    ]
    pH = numpy.asarray(pH)
    o = 0
    for i in range(0, 4):
        o = o - cH[i] * numpy.exp(-(numpy.sum(aH[i, :] * ((L - pH[i, :]) ** 2))))
    return o


def F20(L):
    aH = [
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ]
    aH = numpy.asarray(aH)
    cH = [1, 1.2, 3, 3.2]
    cH = numpy.asarray(cH)
    pH = [
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
    ]
    pH = numpy.asarray(pH)
    o = 0
    for i in range(0, 4):
        o = o - cH[i] * numpy.exp(-(numpy.sum(aH[i, :] * ((L - pH[i, :]) ** 2))))
    return o


def F21(L):
    aSH = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    aSH = numpy.asarray(aSH)
    cSH = numpy.asarray(cSH)
    fit = 0
    for i in range(5):
        v = numpy.matrix(L - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
    o = fit.item(0)
    return o


def F22(L):
    aSH = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    aSH = numpy.asarray(aSH)
    cSH = numpy.asarray(cSH)
    fit = 0
    for i in range(7):
        v = numpy.matrix(L - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
    o = fit.item(0)
    return o


def F23(L):
    aSH = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    aSH = numpy.asarray(aSH)
    cSH = numpy.asarray(cSH)
    fit = 0
    for i in range(10):
        v = numpy.matrix(L - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
    o = fit.item(0)
    return o
"""
def getFunctionDetails(a):
    # [name, lb, ub, dim]
    param = {
        "F1": ["F1", -100, 100, 30],
        "F2": ["F2", -10, 10, 30],
        "F3": ["F3", -100, 100, 30],
        "F4": ["F4", -100, 100, 30],
        "F5": ["F5", -30, 30, 30],
        "F6": ["F6", -100, 100, 30],
        "F7": ["F7", -1.28, 1.28, 30],
        "F8": ["F8", -500, 500, 30],
        "F9": ["F9", -5.12, 5.12, 30],
        "F10": ["F10", -32, 32, 30],
        "F11": ["F11", -600, 600, 30],
        "F12": ["F12", -50, 50, 30],
        "F13": ["F13", -50, 50, 30],
        "F14": ["F14", -65.536, 65.536, 2],
        "F15": ["F15", -5, 5, 4],
        "F16": ["F16", -5, 5, 2],
        "F17": ["F17", -5, 15, 2],
        "F18": ["F18", -2, 2, 2],
        "F19": ["F19", 0, 1, 3],
        "F20": ["F20", 0, 1, 6],
        "F21": ["F21", 0, 10, 4],
        "F22": ["F22", 0, 10, 4],
        "F23": ["F23", 0, 10, 4],
    }
    return param.get(a, "nothing")
"""

# ESAs space mission design benchmarks https://www.esa.int/gsp/ACT/projects/gtop/
from fcmaes.astro import (
    MessFull,
    Messenger,
    Gtoc1,
    Cassini1,
    Cassini2,
    Rosetta,
    Tandem,
    Sagas,
)
def Ca1(x):
    return Cassini1().fun(x)
def Ca2(x):
    return Cassini2().fun(x)
def Ros(x):
    return Rosetta().fun(x)
def Tan(x):
    return Tandem(5).fun(x)
def Sag(x):
    return Sagas().fun(x)
def Mef(x):
    return MessFull().fun(x)
def Mes(x):
    return Messenger().fun(x)
def Gt1(x):
    return Gtoc1().fun(x)

import numpy as np

def cec01(x):
    dim = 4
    p1 = 0.0
    p2 = 0.0
    p3 = 0.0
    d = 72.661
    u = 0.0
    v = 0.0
    wk = 0.0
    pk = 0.0
    m = 32 * dim

    # Calculate u
    for i in range(dim):
        u += x[i] * (1.2)**(dim - i - 1)
    if u < d:
        p1 = (u - d)**2

    # Calculate v
    for i in range(dim):
        v += x[i] * (-1.2)**(dim - i - 1)
    if v < d:
        p2 = (v - d)**2

    # Calculate pk and p3
    for k in range(m + 1):
        for i in range(dim):
            wk += x[i] * ((2 * k / m) - 1)**(dim - i - 1)
        if wk > d:
            pk += (wk - d)**2
        elif wk < d:
            pk += (wk + d)**2
        else:
            pk += 0.0
        wk = 0.0  # Reset wk for the next iteration

    p3 = pk

    # Final output
    o = p1 + p2 + p3
    return o

def cec02(x):
    n = int(np.sqrt(16))  # Compute the square root of 16, assuming `n` is 4
    W = 0.0  # Initialize W to 0
    for i in range(1, n + 1):  # Loop from 1 to n
        xi = x[i - 1]  # Get the i-th element of x
        for k in range(1, n + 1):  # Nested loop from 1 to n
            if i == k:
                I = 1
            else:
                I = 0
            H = 1 / (i + k - 1)  # Calculate H
            Z = xi + (n * (k - 1))  # Calculate Z
            W += abs(H * Z - I)  # Update W with the absolute value of the expression
    return W  # Return the final value of W

def cec03(x):
    n = int(18 / 3)  # Compute n as 18 divided by 3
    d = 0.0  # Initialize d to 0
    total_sum = 0.0  # Initialize sum to 0 (renamed to avoid shadowing built-in `sum`)
    epsilon = 1e-10  # Small constant to prevent division by zero

    for i in range(1, n):  # Loop from 1 to n-1
        xi = x[3 * i - 2]  # Access the (3*i-1)-th element (adjusted for Python indexing)
        for j in range(i + 1, n + 1):  # Loop from i+1 to n
            tmp = 0.0  # Temporary variable for the inner sum
            xj = x[3 * j - 2]  # Access the (3*j-1)-th element (adjusted for Python indexing)
            for k in range(3):  # Loop from 0 to 2
                tmp += (xi + k - 2 - (xj + k - 2))**2  # Update tmp with the squared difference
            d = tmp**3  # Calculate d as tmp^3
            if d > 0:  # Only update if d > 0 to avoid division by zero
                total_sum += (1 / (d + epsilon)**2) - (2 / (d + epsilon))  # Add epsilon to avoid zero

    o = 12.7120622568 + total_sum  # Add the constant value to the total sum
    return o  # Return the result



def cec04(x):
    dim = 10  # Dimension
    total_sum = 0.0  # Initialize sum

    # Shifted and rotated matrices
    shifted_matrix = np.array([43.453613502650342, -75.117860955706732, 54.110917436941946, 2.189362683421635,
                                -3.3813797325740467, -30.849165372014589, 78.077592550813023, -69.901998485392895,
                                37.111456001695004, 52.241020487733664])

    rotated_matrix = np.array([
        [0.8897081082511968, 0.19871231543356224, 0.35531377300377703, 0.0, 0.0, 0.0, 0.0, -0.20660353462835387, 0.0, 0.0],
        [0.10419879983757413, -0.66358499459221376, 0.45164451523757104, 0.0, 0.0, 0.0, 0.0, 0.58720932972857365, 0.0, 0.0],
        [-0.43941933258454113, 0.34165627723133662, 0.81471256710105333, 0.0, 0.0, 0.0, 0.0, -0.16255790164428213, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.42863677588379034, 0.0, -0.63119168084438271, 0.0, 0.0, 0.64642677573936591, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.68398421839009127, 0.0, -0.10280182935971671, 0.0, 0.0, 0.91093354741887778],
        [0.0, 0.0, 0.0, 0.14071437781392207, 0.0, 0.75339632462828454, 0.0, 0.0, 0.64233436924473619, 0.0],
        [0.0, 0.0, 0.0, 0.0, -0.52127926340507358, 0.0, 1.8558504533667268, 0.0, 0.0, -0.12500979819172911],
        [0.066878564394300288, 0.63516876408991430, -0.077542166446725458, 0.0, 0.0, 0.0, 0.0, 0.76555542658372966, 0.0, 0.0],
        [0.0, 0.0, 0.0, -0.89245166717105195, 0.0, -0.18436659152198701, 0.0, 0.0, 0.41175111620270627, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.2215803943548182, 0.0, -0.10421383218771203, 0.0, 0.0, 0.45968278525834394]
    ])

    # Shift and rotate the input vector
    X_shifted = x - shifted_matrix
    shifted_rotated_x = np.dot(X_shifted, rotated_matrix)

    # Compute the function value
    for i in range(dim):
        xi = shifted_rotated_x[i]
        total_sum += xi**2 - 10 * np.cos(2 * np.pi * xi) + 10

    return total_sum

def cec05(x):
    dim = 10
    shifted_matrix = np.array([
        -16.799910337105352, 43.906964270354706, 24.34849185140267,
        -54.897453475230122, 58.499441807390866, 0.11845681821854726,
        70.90374379926536, -0.7779657471822361, 44.729687108066713,
        -68.14877472266032
    ])
    
    rotated_matrix = np.array([
        [-0.7598894912399723, 0.0, 0.0, 0.5979064891770712, 0.0, 0.0, 0.0, -0.25509957135010197, 0.0, 0.0],
        [0.0, -0.06433542223468902, 1.3912090644901491, 0.0, 0.0, 0.0, -0.7035494811301545, 0.0, 0.0, 0.0],
        [0.0, 1.0874018967981698, -0.9362865734577892, 0.0, 0.0, 0.0, -0.22957124131927584, 0.0, 0.0, 0.0],
        [-0.048529941749281533, 0.0, 0.0, 0.33915506135367807, 0.0, 0.0, 0.0, 0.9394778811190786, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -0.7388034758961113, -0.18005470643130739, 0.0, 0.0, -0.07510533003291435, 0.6450650479574964],
        [0.0, 0.0, 0.0, 0.0, 0.22733754217810664, 0.1461061442240787, 0.0, 0.0, 0.8744089942878357, 0.40296345646339404],
        [0.0, -0.9179408195725069, 0.010235511548555627, 0.0, 0.0, 0.0, -0.9353439725277798, 0.0, 0.0, 0.0],
        [0.6482382323319629, 0.0, 0.0, 0.726279336452671, 0.0, 0.0, 0.0, -0.22870399993222773, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -0.09146085004160964, -0.8892250052807102, 0.0, 0.0, 0.31795773407187566, -0.31593778222470387],
        [0.0, 0.0, 0.0, 0.0, -0.6277913497571528, 0.3943503357364224, 0.0, 0.0, 0.35870737301184835, -0.5671815004225745]
    ])
    
    # Shifting
    x = np.array(x) - shifted_matrix
    
    # Rotating
    shifted_rotated_x = np.dot(x, rotated_matrix)
    
    sum_term = 0.0
    multi_term = 1.0
    
    for i in range(dim):
        sum_term += (shifted_rotated_x[i] ** 2) / 4000
        multi_term *= np.cos(shifted_rotated_x[i] / np.sqrt(i + 1))
    
    result = (sum_term - multi_term + 1) + 1
    return result

def cec06(x):
    a = 0.5
    b = 3.0
    kMax = 20
    D = 10
    total_sum = 0.0
    
    shifted_matrix = np.array([
        4.4867071194977996e+01, 8.6557399521842626e-01, -1.2297862364117918e+01, 
        2.9827246270062048e+01, 2.6528060932889602e+01, -6.2879900924339843e+01, 
        -2.2494835379763892e+01, 9.3017723082107295e+00, 1.4887184097844738e+01, 
        -3.1096867523666873e+01
    ])
    
    rotated_matrix = np.array([
        [-1.5433743057196678e-01, 0.0000000000000000e+00, 7.7666311726871273e-01, 0.0000000000000000e+00, 1.1571979400226866e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],
        [0.0000000000000000e+00, 4.6806840267259536e-02, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, -5.9264454599472804e-01, 1.6314935476659614e-01, 7.8737783169590370e-01, 0.0000000000000000e+00, 0.0000000000000000e+00],
        [-1.7410812843826278e+00, 0.0000000000000000e+00, -4.4194799352318298e-01, 0.0000000000000000e+00, 4.4605580480878959e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],
        [0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 2.7077411154472419e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, -8.8999649318267127e-01, 3.6686185770629254e-01],
        [5.4888525059737507e-02, 0.0000000000000000e+00, 1.5570674387300532e+00, 0.0000000000000000e+00, -3.0216546520289828e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],
        [0.0000000000000000e+00, 6.1164921138202333e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, -6.1748299284504526e-01, -1.5999277506278717e-01, -4.6797682388189477e-01, 0.0000000000000000e+00, 0.0000000000000000e+00],
        [0.0000000000000000e+00, -1.1226733726002835e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, -1.3517591002752971e-01, 9.4075663040175728e-01, -2.9000082877106131e-01, 0.0000000000000000e+00, 0.0000000000000000e+00],
        [0.0000000000000000e+00, 7.8172271740335475e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 4.9921405128267116e-01, 2.5052257846765580e-01, 2.7736863877405393e-01, 0.0000000000000000e+00, 0.0000000000000000e+00],
        [0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, -3.0159372777109039e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 2.8348126021733977e-01, 9.1031840499614625e-01],
        [0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 9.1417864987446651e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 3.5713389257830902e-01, 1.9165797370723034e-01]
    ])

    # Shifting
    shifted_x = x - shifted_matrix

    # Rotating
    shifted_rotated_x = np.dot(shifted_x, rotated_matrix)

    for xi in shifted_rotated_x:
        inner_sum_1 = 0.0
        for k in range(kMax + 1):
            inner_sum_1 += (a ** k) * np.cos(2 * np.pi * (b ** k) * (xi + 0.5))
        total_sum += inner_sum_1

    inner_sum_2 = 0.0
    for k in range(kMax + 1):
        inner_sum_2 += (a ** k) * np.cos(np.pi * (b ** k))

    total_sum = total_sum - D * inner_sum_2 + 1
    return total_sum



def cec07(x):
    D = 10
    g = 0.0

    shiftedMatrix = np.array([
        1.5519604466631876, 3.7992270681072, 13.609333677966774, -67.9288744125184,
        79.40774880322056, 46.03413572815904, -64.28081683082544, -47.688475683186425,
        -60.21080731424075, 36.96146955572138
    ])

    rotatedMatrix = np.array([
        [-3.4378315941460673e-02, -7.3911155710735865e-01, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0523399031716010e+00, 0.0, 0.0],
        [1.1485242405257232e+00, 9.9172138327339543e-01, 0.0, 0.0, 0.0, 0.0, 0.0, -9.8221173301823295e-01, 0.0, 0.0],
        [0.0, 0.0, 8.6405702281889096e-01, 0.0, 0.0, -4.9170053952174497e-01, 0.0, 0.0, 0.0, -1.0787048137178114e-01],
        [0.0, 0.0, 0.0, -3.7383444140566896e-01, 3.9203594526066760e-01, 0.0, -7.8796970160513635e-01, 0.0, 2.9267623305420509e-01, 0.0],
        [0.0, 0.0, 0.0, 1.3610747065473336e-01, -6.0804319089793657e-01, 0.0, -5.6714757040795583e-01, 0.0, -5.3861105430076284e-01, 0.0],
        [0.0, 0.0, 8.5194811610211113e-02, 0.0, 0.0, -6.8358420399848130e-02, 0.0, 0.0, 0.0, 9.9401658458757081e-01],
        [0.0, 0.0, 0.0, -8.5125893429894595e-01, 8.0910637486172887e-03, 0.0, 2.3333420826786389e-01, 0.0, -4.6994458047268511e-01, 0.0],
        [1.2482031580671493e+00, -4.3601697061224165e-01, 0.0, 0.0, 0.0, 0.0, 0.0, -1.7452756850803894e-01, 0.0, 0.0],
        [0.0, 0.0, 0.0, 3.4217070831253948e-01, 6.9030850372398578e-01, 0.0, -5.4795346377890963e-02, 0.0, -6.3513057403542883e-01, 0.0],
        [0.0, 0.0, 4.9613234664961658e-01, 0.0, 0.0, 8.6807701605010890e-01, 0.0, 0.0, 0.0, 1.7175238382100888e-02]
    ])

    # Shifting
    shiftedX = x - shiftedMatrix

    # Rotating
    shiftedRotatedX = np.dot(shiftedX, rotatedMatrix)

    for i in range(D):
        xi = shiftedRotatedX[i]
        zi = xi + 420.9687462275036
        if abs(zi) <= 500:
            g += zi * np.sin(np.sqrt(abs(zi)))
        elif zi > 500:
            g += ((500 - (zi % 500)) * np.sin(np.sqrt(abs(500 - (zi % 500)))) - ((zi - 500) ** 2) / (10000 * D))
        elif zi < -500:
            g += (((zi % 500) - 500) * np.sin(np.sqrt(abs((zi % 500) - 500))) - ((zi - 500) ** 2) / (10000 * D))

    o = (418.9829 * D - g)
    return o

def cec08(x):
    D = 10
    g = 0.0

    # Shifted matrix
    shifted_matrix = np.array([
        75.8095362017907, 50.8749434961355, 15.175339549395872, 11.931806696547099, 57.875148867198789,
        67.627011010249618, -32.825950734701912, -25.75399813510198, -47.44665665898782, 4.041532391701594
    ])

    # Rotated matrix
    rotated_matrix = np.array([
        [ 0.32765524541169905,  0.0,  0.9493315755326415,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.6250327349341648],
        [ 0.0, -0.3941668928110215,  0.0,  0.0,  0.6486658379192489,  0.0, -0.6148405667174165,  0.2140938318747867,  0.0,  0.0],
        [ 0.8966435770857878,  0.0,  0.5375307662270435,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.7220887004986716],
        [ 0.0,  0.0,  0.0, -0.1733434047971036,  0.0,  0.1783302480945321,  0.0,  0.0, -0.9685816365324094,  0.0],
        [ 0.0,  0.5540963793461997,  0.0,  0.0, -0.3025040233918769,  0.0, -0.4559073871465514,  0.627389012154631,  0.0,  0.0],
        [ 0.0,  0.0,  0.0, -0.9674651764079433,  0.0,  0.1531975677515603,  0.0,  0.0,  0.20134954102999447,  0.0],
        [ 0.0,  0.6259454530374925,  0.0,  0.0,  0.6975032106729394,  0.0,  0.3470026747562117,  0.03564694425423177,  0.0,  0.0],
        [ 0.0,  0.3818402189775527,  0.0,  0.0, -0.03483127401038803,  0.0, -0.5419489603051616, -0.7478476809793169,  0.0,  0.0],
        [ 0.0,  0.0,  0.0, -0.18429106449108246,  0.0, -0.9719716188498707,  0.0,  0.0, -0.14597251693095872,  0.0],
        [ 1.715366551364125,  0.0, -0.7894470726378124,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.3003461621151572]
    ])

    # Shifting
    shifted_x = x - shifted_matrix

    # Rotating
    shifted_rotated_x = np.dot(shifted_x, rotated_matrix)

    # Calculating the objective function
    for i in range(D - 1):
        j = i + 1
        xi = shifted_rotated_x[i]
        yi = shifted_rotated_x[j]
        g += 0.5 + (np.sin(np.sqrt(xi**2 + yi**2))**2 - 0.5) / ((1 + 0.001 * (xi**2 + yi**2))**2)

    xi = shifted_rotated_x[D - 1]
    yi = shifted_rotated_x[0]
    g += 0.5 + (np.sin(np.sqrt(xi**2 + yi**2))**2 - 0.5) / ((1 + 0.001 * (xi**2 + yi**2))**2)

    return g + 1


def cec09(x):
    D = 10
    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0
    
    shifted_matrix = np.array([-6.0107960952496171e+00, -6.3449972860258995e+01, -3.6938623728667750e+00, 
                               -2.7449007717635965e+00, -5.3547271030744199e+01, 3.1015786282259867e+01, 
                               2.3200459416583499e+00, -4.6987858548289097e+01, 3.5061378905112562e+01, 
                               -3.4047417731046465e+00])
    
    rotated_matrix = np.array([
        [-7.6923624057192400e-02,  0.0000000000000000e+00,  7.2809258658661558e-02,  6.1371429917067155e-01,  0.0000000000000000e+00,  7.8239141541106805e-01,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00],
        [0.0000000000000000e+00, -1.1499983823069659e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  1.5729072158274271e-01,  0.0000000000000000e+00,  0.0000000000000000e+00, -1.3309066870600375e+00,  0.0000000000000000e+00,  0.0000000000000000e+00],
        [-1.6730831752378217e-02,  0.0000000000000000e+00,  4.9480374519689890e-01,  6.5982384537901573e-01,  0.0000000000000000e+00, -5.6526261691115431e-01,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00],
        [9.1421044415115027e-01,  0.0000000000000000e+00, -3.3249140365486585e-01,  2.2489758522716782e-01,  0.0000000000000000e+00, -5.5586027556918202e-02,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00],
        [0.0000000000000000e+00, -1.2704704488967578e+00,  0.0000000000000000e+00,  0.0000000000000000e+00, -7.6341623484218024e-01,  0.0000000000000000e+00,  0.0000000000000000e+00,  4.9980922801223232e-01,  0.0000000000000000e+00,  0.0000000000000000e+00],
        [-3.9751993551989051e-01,  0.0000000000000000e+00, -7.9957334378299227e-01,  3.7068629354440513e-01,  0.0000000000000000e+00, -2.5544478964007222e-01,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00],
        [0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  9.0184624725396623e-02,  0.0000000000000000e+00, -3.4243496198122719e-01, -9.3520320266563195e-01],
        [0.0000000000000000e+00,  9.7696981452382070e-01,  0.0000000000000000e+00,  0.0000000000000000e+00, -6.8376531090322690e-01,  0.0000000000000000e+00,  0.0000000000000000e+00, -4.7094671586086240e-01,  0.0000000000000000e+00,  0.0000000000000000e+00],
        [0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00, -6.1661688294908079e-01,  0.0000000000000000e+00, -7.5660067900409822e-01,  2.1757534831110126e-01],
        [0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00, -7.8208078427058847e-01,  0.0000000000000000e+00,  5.5704013261474550e-01, -2.7938492717261593e-01]
    ])
    
    # Shifting
    shifted_x = x - shifted_matrix
    
    # Rotating
    shifted_rotated_x = np.dot(shifted_x, rotated_matrix)
    
    for i in range(D):
        xi = shifted_rotated_x[i]
        sum1 += xi**2 - D
        sum2 += xi**2
        sum3 += xi

    result = (abs(sum1)**(1/4)) + (0.5 * sum2 + sum3) / D + 0.5
    return result


def cec10(x):
    D = 10
    shiftedMatrix = np.array([6.1441309549566370e-001, 1.8049534213689469e+001, 5.1107558757100151e+001,
                             5.1022671188681272e+000, -4.7667984552250942e+001, -7.3770454911164904e+000,
                             -1.1534252828772665e+001, 7.4568439937919834e+001, 1.9208808661355789e+001,
                             3.1262392306880571e+001])
    rotatedMatrix = np.array([[-3.6144665808053256e-02, 0.0000000000000000e+00, 0.0000000000000000e+00,
                               -1.0275628429515489e-01, 0.0000000000000000e+00, 0.0000000000000000e+00,
                               -9.9404965126067890e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],
                              [0.0000000000000000e+00, 5.7732032557209267e-01, 6.3720045355332378e-01,
                               0.0000000000000000e+00, -1.2849837048835189e-01, 0.0000000000000000e+00,
                               0.0000000000000000e+00, 4.9413054191641514e-01, 0.0000000000000000e+00,
                               0.0000000000000000e+00],
                              [0.0000000000000000e+00, -3.9586712917636224e-01, -2.8968310271397846e-01,
                               0.0000000000000000e+00, -5.1670886261404880e-01, 0.0000000000000000e+00,
                               0.0000000000000000e+00, 7.0170140895951061e-01, 0.0000000000000000e+00,
                               0.0000000000000000e+00],
                              [8.7364129527216294e-01, 0.0000000000000000e+00, 0.0000000000000000e+00,
                               4.7972086095511046e-01, 0.0000000000000000e+00, 0.0000000000000000e+00,
                               -8.1355901812131204e-02, 0.0000000000000000e+00, 0.0000000000000000e+00,
                               0.0000000000000000e+00],
                              [0.0000000000000000e+00, 7.1410594914990388e-01, -6.7354832197676640e-01,
                               0.0000000000000000e+00, -1.9014210237853424e-01, 0.0000000000000000e+00,
                               0.0000000000000000e+00, -1.5209610582332084e-02, 0.0000000000000000e+00,
                               0.0000000000000000e+00],
                              [0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
                               0.0000000000000000e+00, 0.0000000000000000e+00, -2.7911553415074425e-01,
                               0.0000000000000000e+00, 0.0000000000000000e+00, 1.6909147808954352e+00,
                               -5.7964733149576031e-01],
                              [-4.8522618471059614e-01, 0.0000000000000000e+00, 0.0000000000000000e+00,
                               8.7138340677473369e-01, 0.0000000000000000e+00, 0.0000000000000000e+00,
                               -7.2432783108600157e-02, 0.0000000000000000e+00, 0.0000000000000000e+00,
                               0.0000000000000000e+00],
                              [0.0000000000000000e+00, -6.5689502748026013e-03, 2.3746987168001482e-01,
                               0.0000000000000000e+00, -8.2483095297218667e-01, 0.0000000000000000e+00,
                               0.0000000000000000e+00, -5.1304854346890183e-01, 0.0000000000000000e+00,
                               0.0000000000000000e+00],
                              [0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
                               0.0000000000000000e+00, 0.0000000000000000e+00, -1.3268067876803948e+00,
                               0.0000000000000000e+00, 0.0000000000000000e+00, 7.1252154299165178e-04,
                               -8.8929149979214617e-01],
                              [0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
                               0.0000000000000000e+00, 0.0000000000000000e+00, -6.6352395365628714e-01,
                               0.0000000000000000e+00, 0.0000000000000000e+00, 4.6075742954200716e-01,
                               6.9770224192342367e-01]])

    # Shifting
    shiftedX = x - shiftedMatrix

    # Rotating
    shiftedRotatedX = np.dot(shiftedX, rotatedMatrix)

    # Compute the result
    result = -20 * np.exp(-0.2 * np.sqrt(np.sum(shiftedRotatedX**2) / D)) - \
            np.exp(np.sum(np.cos(2 * np.pi * shiftedRotatedX)) / D) + 20 + np.exp(1)
    
    return result

def getFunctionDetails(a):
    # [name, lb, ub, dim]
    param = {
        "F1": ["F1", -100, 100, 30],
        "F2": ["F2", -10, 10, 30],
        "F3": ["F3", -100, 100, 30],
        "F4": ["F4", -100, 100, 30],
        "F5": ["F5", -30, 30, 30],
        "F6": ["F6", -100, 100, 30],
        "F7": ["F7", -1.28, 1.28, 30],
        "F8": ["F8", -500, 500, 30],
        "F9": ["F9", -5.12, 5.12, 30],
        "F10": ["F10", -32, 32, 30],
        "F11": ["F11", -600, 600, 30],
        "F12": ["F12", -50, 50, 30],
        "F13": ["F13", -50, 50, 30],
        "F14": ["F14", -65.536, 65.536, 2],
        "F15": ["F15", -5, 5, 4],
        "F16": ["F16", -5, 5, 2],
        "F17": ["F17", -5, 15, 2],
        "F18": ["F18", -2, 2, 2],
        "F19": ["F19", 0, 1, 3],
        "F20": ["F20", 0, 1, 6],
        "F21": ["F21", 0, 10, 4],
        "F22": ["F22", 0, 10, 4],
        "F23": ["F23", 0, 10, 4],
        "cec01":["cec01", -8192, 8192, 9],
        "cec02":["cec02", -16384, 16384, 16],
        "cec03":["cec03", -4, 4, 18],
        "cec04":["cec04", -100, 100, 10],
        "cec05":["cec05", -100, 100, 10],
        "cec06":["cec06", -100, 100, 10],
        "cec07":["cec07", -100, 100, 10],
        "cec08":["cec08", -100, 100, 10],
        "cec09":["cec09", -100, 100, 10],
        "cec10":["cec10", -100, 100, 10],
        "Ca1": [
            "Ca1",
            Cassini1().bounds.lb,
            Cassini1().bounds.ub,
            len(Cassini1().bounds.lb),
        ],
        "Ca2": [
            "Ca2",
            Cassini2().bounds.lb,
            Cassini2().bounds.ub,
            len(Cassini2().bounds.lb),
        ],
        "Gt1": ["Gt1", Gtoc1().bounds.lb, Gtoc1().bounds.ub, len(Gtoc1().bounds.lb)],
        "Mes": [
            "Mes",
            Messenger().bounds.lb,
            Messenger().bounds.ub,
            len(Messenger().bounds.lb),
        ],
        "Mef": [
            "Mef",
            MessFull().bounds.lb,
            MessFull().bounds.ub,
            len(MessFull().bounds.lb),
        ],
        "Sag": ["Sag", Sagas().bounds.lb, Sagas().bounds.ub, len(Sagas().bounds.lb)],
        "Tan": [
            "Tan",
            Tandem(5).bounds.lb,
            Tandem(5).bounds.ub,
            len(Tandem(5).bounds.lb),
        ],
        "Ros": [
            "Ros",
            Rosetta().bounds.lb,
            Rosetta().bounds.ub,
            len(Rosetta().bounds.lb),
        ],
    }
    return param.get(a, "nothing")
