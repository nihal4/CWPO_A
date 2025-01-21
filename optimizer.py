# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:50:25 2016

@author: hossam
"""
from pathlib import Path
import optimizers.PSO as pso
import optimizers.MVO as mvo
import optimizers.CWPO as CWPO
import optimizers.GWO as gwo
import optimizers.MFO as mfo
import optimizers.CS as cs
import optimizers.BAT as bat
import optimizers.WOA as woa
import optimizers.FFA as ffa
import optimizers.SSA as ssa
import optimizers.GA as ga
import optimizers.HHO as hho
import optimizers.SCA as sca
import optimizers.JAYA as jaya
import optimizers.DE as de
import optimizers.L_SHADE as lshade
import optimizers.COA as coa
import optimizers.ROA as roa
import optimizers.GGO as ggo
import optimizers.SO as so
import optimizers.AO as ao
import benchmarks
import csv
import numpy
import time
import warnings
import os
import plot_convergence as conv_plot
import plot_boxplot as box_plot

warnings.simplefilter(action="ignore")


def selector(algo, func_details, popSize, Iter):
    function_name = func_details[0]
    lb = func_details[1]
    ub = func_details[2]
    dim = func_details[3]

    if algo == "SSA":
        x = ssa.SSA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "PSO":
        x = pso.PSO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "GA":
        x = ga.GA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "BAT":
        x = bat.BAT(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "FFA":
        x = ffa.FFA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "CWPO":
        x = CWPO.CWPO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "WOA":
        x = woa.WOA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "MVO":
        x = mvo.MVO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "MFO":
        x = mfo.MFO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "CS":
        x = cs.CS(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "HHO":
        x = hho.HHO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "SCA":
        x = sca.SCA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "JAYA":
        x = jaya.JAYA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "DE":
        x = de.DE(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "GWO":
        x = gwo.GWO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "L_SHADE":
        x = lshade.L_SHADE(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "COA":
        x = coa.COA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "ROA":
        x = roa.ROA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "GGO":
        x = ggo.GGO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "SO":
        x = so.SO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "AO":
        x = ao.AO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    else:
        return null
    return x


def run(optimizer, objectivefunc, NumOfRuns, params, export_flags):
    """
    Main interface of the framework for running experiments.

    Parameters
    ----------
    optimizer : list
        The list of optimizer names
    objectivefunc : list
        The list of benchmark functions
    NumOfRuns : int
        The number of independent runs
    params : dict
        The set of parameters:
            1. Size of population (PopulationSize)
            2. Number of iterations (Iterations)
    export_flags : dict
        The set of Boolean flags:
            1. Export_avg (Export the average results to a file)
            2. Export_details (Export detailed results to files)
            3. Export_convergence (Export convergence plots)
            4. Export_boxplot (Export box plots)

    Returns
    -------
    None
    """

    # General parameters for all optimizers
    PopulationSize = params["PopulationSize"]
    Iterations = params["Iterations"]

    # Export flags
    Export = export_flags["Export_avg"]
    Export_details = export_flags["Export_details"]
    Export_convergence = export_flags["Export_convergence"]
    Export_boxplot = export_flags["Export_boxplot"]

    Flag = False
    Flag_details = False

    # CSV Header for the convergence
    CnvgHeader = [f"Iter{l + 1}" for l in range(Iterations)]

    results_directory = time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
    Path(results_directory).mkdir(parents=True, exist_ok=True)

    for optimizer_name in optimizer:
        for objective in objectivefunc:
            convergence = [[] for _ in range(NumOfRuns)]  # Initialize as list of lists
            executionTime = [0] * NumOfRuns

            for run_idx in range(NumOfRuns):
                func_details = benchmarks.getFunctionDetails(objective)
                x = selector(optimizer_name, func_details, PopulationSize, Iterations)
                convergence[run_idx] = x.convergence  # Populate convergence for each run
                optimizerName = x.optimizer
                objfname = x.objfname

                # Export detailed results
                if Export_details:
                    ExportToFile = results_directory + "experiment_details.csv"
                    with open(ExportToFile, "a", newline="") as out:
                        writer = csv.writer(out, delimiter=",")
                        if not Flag_details:  # Write the header once
                            header = ["Optimizer", "objfname", "ExecutionTime", "Individual"] + CnvgHeader
                            writer.writerow(header)
                            Flag_details = True

                        executionTime[run_idx] = x.executionTime
                        row = [x.optimizer, x.objfname, x.executionTime, x.bestIndividual] + x.convergence.tolist()
                        writer.writerow(row)

            # Export summary results
            if Export:
                ExportToFile = results_directory + "experiment.csv"
                with open(ExportToFile, "a", newline="") as out:
                    writer = csv.writer(out, delimiter=",")
                    if not Flag:  # Write the header once
                        header = ["Optimizer", "objfname", "Avg_Min", "Std_Min"]
                        writer.writerow(header)
                        Flag = True

                    min_values = [min(run) for run in convergence if len(run) > 0]  # Ensure non-empty lists
                    avg_min = numpy.mean(min_values)
                    std_min = numpy.std(min_values)
                    writer.writerow([optimizerName, objfname, avg_min, std_min])

    # Optional: Export convergence and box plots
    if Export_convergence:
        conv_plot.run(results_directory, optimizer, objectivefunc, Iterations)

    if Export_boxplot:
        box_plot.run(results_directory, optimizer, objectivefunc, Iterations)

    if not Flag:  # No experiment was executed
        print("No optimizer or cost function selected. Check available optimizers and cost functions.")

    print("Execution completed.")

