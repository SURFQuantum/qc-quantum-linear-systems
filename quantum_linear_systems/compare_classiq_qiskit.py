"""Compare the Classiq and Qiskit implementations on different use-cases and plot the results."""
from typing import List
from datetime import datetime

import csv
import numpy as np
import matplotlib.pyplot as plt

from quantum_linear_systems.toymodels import (Qiskit4QubitExample,
                                              ClassiqDemoExample,
                                              VolterraProblem,
                                              ToyModel,
                                              ScalingTestModel
                                              )
from quantum_linear_systems.quantum_linear_solver import QuantumLinearSolver
from quantum_linear_systems.utils import relative_distance_quantum_classical_solution
from quantum_linear_systems.plotting import plot_compare_csol_vs_qsol, plot_depth_runtime_distance_vs_problem


def append_to_csv(filename, data):
    """Helper function to append data to an existing csv file."""
    try:
        with open(filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(data)
    except Exception as exception:
        print(f"Error while appending to CSV: {exception}")


def solve_models(models, method, save_file):
    """
    Solve a set of quantum models using the given solver function.

    Parameters:
        models (list): A list of quantum models to be solved.
        method (str): Which method to use for solving the model.

    Returns:
        tuple: A tuple containing the following elements:
            - quantum_solutions (list): List of quantum solutions for each model.
            - classical_solutions (list): List of classical solutions for each model.
            - performance_data (tuple): A nested tuple containing performance metrics for each model:
                - depths (list): List of depths for each model's solution.
                - run_times (list): List of run times for each model's solution.
                - rel_distances (list): List of relative distances between quantum and classical solutions.
    """
    quantum_solutions, classical_solutions, run_times, depths, rel_distances = [], [], [], [], []

    for model in models:
        print(datetime.now().strftime("%H:%M:%S"))
        print(f"Solving {model.name} using {method}.")
        qls = QuantumLinearSolver()
        qsol = qls.solve(matrix_a=model.matrix_a, vector_b=model.vector_b, method=method, file_basename=None)

        # todo: once all normalizations are fixed this should be unnecessary, for plotting its still cool probably
        qsol = qsol / np.linalg.norm(qsol)
        csol = model.classical_solution / np.linalg.norm(model.classical_solution)
        rel_dis = relative_distance_quantum_classical_solution(quantum_solution=qsol,
                                                               classical_solution=csol)
        quantum_solutions.append(qsol)
        classical_solutions.append(csol)
        run_times.append(qls.run_time)
        depths.append(qls.circuit_depth)
        rel_distances.append(rel_dis)

        data = [model.name, method, qsol, csol, qls.circuit_depth, qls.run_time, rel_dis]
        append_to_csv(filename=save_file, data=data)

    return quantum_solutions, classical_solutions, (depths, run_times, rel_distances)


def compare_qls_and_plot(
        models: List[ToyModel],
        qiskit: bool = True,
        classiq: bool = True,
        filebasename: str = "comparison"
) -> None:
    """Compare different implementations of quantum linear solvers and plot the results"""
    filename = f"{filebasename}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    headers = ['model_name', 'solver', 'quantum_solution', 'classical_solution', 'circuit_depth',
               'run_time', 'relative_distance']

    with open(filename, mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

    if classiq:
        # classiq first because if something fails its classiq
        qsols_c_hhl, csols, drd_c_hhl = solve_models(models=models, method="hhl_classiq", save_file=filename)
        # qsols_c_vqls, _, drd_c_vqls = solve_models(models=models, method="vqls_classiq", save_file=filename)
    if qiskit:
        qsols_q_hhl, csols, drd_q_hhl = solve_models(models=models, method="hhl_qiskit", save_file=filename)
        qsols_q_vqls, _, drd_q_vqls = solve_models(models=models, method="vqls_qiskit", save_file=filename)

    n_problems = len(models)

    # Create subplots for each problem
    _, axs = plt.subplots(n_problems, 2, figsize=(12, 6 * n_problems))

    for i in range(n_problems):
        qmn = []
        if classiq:
            qmn.append((qsols_c_hhl[i], "go", "hhl_classiq"))
            # qmn.append((qsols_c_vqls[i], "go", "vqls_classiq"))
        if qiskit:
            qmn.append((qsols_q_hhl[i], "r^", "hhl_qiskit"))
            qmn.append((qsols_q_vqls[i], "kx", "vqls_qiskit"))

        plot_compare_csol_vs_qsol(classical_solution=csols[i], qsols_marker_name=qmn,
                                  title=f"Statevectors: {models[i].name}", axis=axs[i, 0])
    drdmn = []
    if classiq:
        drdmn.append(drd_c_hhl + ("go", "hhl_classiq"))
        # drdmn.append(drd_c_vqls + ("go", "vqls_classiq"))
    if qiskit:
        drdmn.append(drd_q_hhl + ("r^", "hhl_qiskit"))
        drdmn.append(drd_q_vqls + ("kx", "vqls_qiskit"))

    plot_depth_runtime_distance_vs_problem(depth_runtime_distance_marker_name=drdmn, problems=models, axs=axs[:, 1])

    plt.tight_layout()
    plt.savefig(f'comparison_plot_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    # matrix test:

    toymodels = [ClassiqDemoExample(), Qiskit4QubitExample(), VolterraProblem(num_qubits=2)]
    # VolterraProblem(num_qubits=3)]
    # toymodels = [ClassiqDemoExample(), Qiskit4QubitExample(), VolterraProblem(num_qubits=2),
    #              SimpleHamiltonianModel(3, 2),
    #              SimpleHamiltonianModel(3, 3)]

    # Note classiq can't solve n>=3 here classiq.exceptions.ClassiqAPIError: Error number 73900 occurred.
    # The exponentiation constraints are not satisfiable. Minimal max_depth is 1184.
    # Qiskit has no problem and solves it rather quickly.

    compare_qls_and_plot(models=toymodels, qiskit=True, classiq=True)

    matrix_size_models = []
    for n in range(2, 8):
        matrix_size_models.append(ScalingTestModel(matrix_size=n))
    matrix_sparsity_models = []
    for s in range(1, 9):
        matrix_sparsity_models.append(ScalingTestModel(matrix_size=8, matrix_s=s))
