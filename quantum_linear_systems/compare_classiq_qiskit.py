"""Compare the Classiq and Qiskit implementations on different use-cases and plot the results."""
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes


from quantum_linear_systems.toymodels import (Qiskit4QubitExample,
                                              ClassiqDemoExample,
                                              VolterraProblem)
from quantum_linear_systems.hhl_qiskit_implementation import qiskit_hhl
from quantum_linear_systems.hhl_classiq_implementation import classiq_hhl
from quantum_linear_systems.vqls_qiskit_implementation import qiskit_vqls
from quantum_linear_systems.utils import relative_distance_quantum_classical_solution
from quantum_linear_systems.plotting import plot_compare_csol_vs_qsol, plot_depth_runtime_distance_vs_problem


def solve_models(solver_function, models, needs_ansatz=False):
    """
    Solve a set of quantum models using the given solver function.

    Parameters:
        solver_function (callable): A function that solves a quantum model and returns
            the quantum and classical solutions, along with other relevant information.
        models (list): A list of quantum models to be solved.
        needs_ansatz (bool, optional): Whether the solver function requires an ansatz as input.
            Default is False.

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
        if not needs_ansatz:
            qsol, csol, depth, _, run_time = solver_function(model=model, show_circuit=False)
        else:
            ansatz = RealAmplitudes(num_qubits=int(np.log2(model.matrix_a.shape[0])),
                                    entanglement="full", reps=3, insert_barriers=False)
            qsol, csol, depth, _, run_time = solver_function(model=model, ansatz=ansatz, show_circuit=False)
        rel_dis = relative_distance_quantum_classical_solution(quantum_solution=qsol, classical_solution=csol)
        quantum_solutions.append(qsol)
        classical_solutions.append(csol)
        run_times.append(run_time)
        depths.append(depth)
        rel_distances.append(rel_dis)

    return quantum_solutions, classical_solutions, (depths, run_times, rel_distances)


if __name__ == "__main__":
    # toymodels = [ClassiqDemoExample(), Qiskit4QubitExample(), VolterraProblem(num_qubits=2),
    #              VolterraProblem(num_qubits=3)]
    toymodels = [ClassiqDemoExample(), Qiskit4QubitExample(), VolterraProblem(num_qubits=2)]
    # Note classiq can't solve n>=3 here classiq.exceptions.ClassiqAPIError: Error number 73900 occurred.
    # The exponentiation constraints are not satisfiable. Minimal max_depth is 1184.
    # Qiskit has no problem and solves it rather quickly.

    qsols_q_hhl, csols, depth_runtime_distance_q_hhl = solve_models(solver_function=qiskit_hhl, models=toymodels)
    qsols_q_vqls, _, depth_runtime_distance_q_vqls = solve_models(solver_function=qiskit_vqls, models=toymodels,
                                                                  needs_ansatz=True)
    qsols_c_hhl, _, depth_runtime_distance_c_hhl = solve_models(solver_function=classiq_hhl, models=toymodels)
    # qsols_c_vqls, _, depth_runtime_distance_c_vqls = (
    #     solve_models(solver_function=classiq_vqls, models=toymodels, ansatz=vqls_ansatz))

    N_PROBLEMS = len(toymodels)

    # Create subplots for each problem
    fig, axs = plt.subplots(N_PROBLEMS, 2, figsize=(12, 6 * N_PROBLEMS))

    for i in range(N_PROBLEMS):
        qmn = [
            (qsols_c_hhl[i], "go", "hhl_classiq"),
            (qsols_q_hhl[i], "r^", "hhl_qiskit"),
            # (qsols_c_vqls[i], "go", "vqls_classiq"),
            (qsols_q_vqls[i], "kx", "vqls_qiskit"),
        ]
        plot_compare_csol_vs_qsol(classical_solution=csols[i], qsols_marker_name=qmn,
                                  title=f"Statevectors: {toymodels[i].name}", axis=axs[i, 0])
    drdmn = [
        depth_runtime_distance_c_hhl + ("go", "hhl_classiq"),
        depth_runtime_distance_q_hhl + ("r^", "hhl_qiskit"),
        # depth_runtime_distance_c_vqls + ("go", "vqls_classiq"),
        depth_runtime_distance_q_vqls + ("kx", "vqls_qiskit"),
    ]
    plot_depth_runtime_distance_vs_problem(depth_runtime_distance_marker_name=drdmn, problems=toymodels, axs=axs[:, 1])

    plt.tight_layout()
    plt.savefig(f'comparison_plot_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png', dpi=300)
    plt.show()
