"""Utility functions that can be imported by either implementation."""
import numpy as np
import matplotlib.pyplot as plt


def make_matrix_hermitian(matrix):
    """Creates a hermitian version of a NxM :obj:np.array A as a  (N+M)x(N+M) block matrix [[0 ,A], [A_dagger, 0]]."""
    shape = matrix.shape
    upper_zero = np.zeros(shape=(shape[0], shape[0]))
    lower_zero = np.zeros(shape=(shape[1], shape[1]))
    matrix_dagger = matrix.conj().T
    hermitian_matrix = np.block([[upper_zero, matrix], [matrix_dagger, lower_zero]])
    assert np.array_equal(hermitian_matrix, hermitian_matrix.conj().T)
    return hermitian_matrix


def expand_b_vector(unexpanded_vector, non_hermitian_matrix):
    """Expand vector according to the expansion of the matrix to make it hermitian b -> (b 0)."""
    shape = non_hermitian_matrix.shape
    lower_zero = np.zeros(shape=(shape[1], 1))
    return np.block([[unexpanded_vector], [lower_zero]])


def extract_x_from_expanded(expanded_solution_vector: np.array, non_hermitian_matrix: np.array = None):
    """The expanded problem returns a vector y=(0 x), this function returns x from input y."""
    if non_hermitian_matrix is not None:
        index = non_hermitian_matrix.shape[0]
    else:
        index = int(expanded_solution_vector.flatten().shape[0] / 2)
    return expanded_solution_vector[index:].flatten()


def extract_hhl_solution_vector_from_state_vector(hermitian_matrix: np.array, state_vector: np.array):
    """Extract the solution vector x from the full state vector of the HHL problem which also includes 1 aux. qubit and
    multiple work qubits encoding the eigenvalues.
    """
    size_of_hermitian_matrix = hermitian_matrix.shape[1]
    number_of_qubits_in_result = int(np.log2(len(state_vector)))
    binary_rep = "1" + (number_of_qubits_in_result-1) * "0"
    not_normalized_vec = np.real(state_vector[int(binary_rep, 2):(int(binary_rep, 2) + size_of_hermitian_matrix)])

    return not_normalized_vec / np.linalg.norm(not_normalized_vec)


def relative_distance_quantum_classical_solution(quantum_solution: np.ndarray, classical_solution: np.ndarray) -> float:
    """Calculate relative distance of quantum and classical solutions in percent."""
    return np.linalg.norm(classical_solution - quantum_solution) / np.linalg.norm(classical_solution) * 100


def plot_csol_vs_qsol(classical_solution: np.ndarray, quantum_solution: np.ndarray, title: str) -> None:
    """
    Plot classical and quantum solution vectors side by side.

    Parameters:
        classical_solution (numpy.ndarray): Array representing the classical solution.
        quantum_solution (numpy.ndarray): Array representing the quantum solution.
        title (str): Title for the plot.
    """
    _, axis = plt.subplots()

    axis.plot(classical_solution, "bs", label="classical")
    axis.plot(quantum_solution, "ro", label="HHL")
    axis.legend()
    axis.set_xlabel("$i$")
    axis.set_ylabel("$x_i$")
    axis.set_ylim(0, 1)
    axis.set_title(title)
    axis.grid(True)
    plt.show()


def plot_compare_csol_vs_qsol(classical_solution: np.ndarray, quantum_solution_classiq: np.ndarray,
                              quantum_solution_qiskit: np.ndarray, title: str, axis=None) -> None:
    """
    Plot classical and quantum solutions side by side.

    Parameters:
        classical_solution (numpy.ndarray): Array representing the classical solution.
        quantum_solution_classiq (numpy.ndarray): Array representing the quantum solution output by classiq.
        quantum_solution_qiskit (numpy.ndarray): Array representing the quantum solution output by qiskit.
        title (str): Title for the plot.
        axis (matplotlib.axes._subplots.AxesSubplot): Axes to use for the plot. If None, a new subplot will be created.
    """
    if axis is None:
        fig, axis = plt.subplots()

    axis.plot(classical_solution, "bs", label="classical")
    axis.plot(quantum_solution_classiq, "go", label="HHL_classiq")
    axis.plot(quantum_solution_qiskit, "r^", label="HHL_qiskit")
    axis.legend()
    axis.set_xlabel("$i$")
    axis.set_ylabel("$x_i$")
    axis.set_ylim(0, 1)
    axis.set_title(title)
    axis.grid(True)


def plot_depth_runtime_distance_vs_problem(
    depth_classiq: list, depth_qiskit: list, runtime_classiq: list, runtime_qiskit: list, distance_classiq: list,
        distance_qiskit: list, problems: list, axs: list = None
) -> None:
    """
    Plot depth and runtime of two algorithms side by side for each problem index.

    Parameters:
        depth_classiq (list): List of circuit depths for classiq for each problem.
        depth_qiskit (list): List of circuit depths for qiskit for each problem.
        runtime_classiq (list): List of run_times for classiq for each problem.
        runtime_qiskit (list): List of run_times for qiskit for each problem.
        distance_classiq (list): List of relative distances of quantum/classical solutions for classiq for each problem.
        distance_qiskit (list): List of relative distances of quantum/classical solutions for qiskit for each problem.
        problems (list): List of problem objects with a 'name' attribute.
        axs (list): List of axes to use for the plot. If None, a new subplot will be created.
    """
    problem_names = [problem.name for problem in problems]

    if axs is None or len(axs) < 3:
        _, axs = plt.subplots(3, 1, figsize=(10, 6))

    axs[0].plot(problem_names, depth_classiq, "go", label="Classiq")
    axs[0].plot(problem_names, depth_qiskit, "r^", label="Qiskit")
    axs[0].legend()
    axs[0].set_ylabel("Circuit Depth")
    axs[0].set_title("Circuit Depth Comparison")
    axs[0].grid(True)

    axs[1].plot(problem_names, runtime_classiq, "go", label="Classiq")
    axs[1].plot(problem_names, runtime_qiskit, "r^", label="Qiskit")
    axs[1].legend()
    axs[1].set_ylabel("Run Time [s]")
    axs[1].set_title("Run Time Comparison")
    axs[1].grid(True)

    axs[2].plot(problem_names, distance_classiq, "go", label="Classiq")
    axs[2].plot(problem_names, distance_qiskit, "r^", label="Qiskit")
    axs[2].legend()
    axs[2].set_ylabel("Relative Distance [%]")
    axs[2].set_title("Rel. Distance of Quantum/Classical Solution")
    axs[2].grid(True)


def print_results(quantum_solution: np.ndarray, classical_solution: np.ndarray, run_time: float, name: str,
                  plot: bool = True) -> None:
    """
    Print results of classical and quantum solutions and optionally plot them.

    Parameters:
        quantum_solution (numpy.ndarray): Quantum solution.
        classical_solution (numpy.ndarray): Classical solution.
        run_time (float): Time taken for the computation.
        name (str): Name of the solution.
        plot (bool, optional): Whether to generate and display a plot. Default is True.
    """
    # todo: decide whether or not to work with normalization here
    classical_solution /= np.linalg.norm(classical_solution)
    quantum_solution /= np.linalg.norm(quantum_solution)
    print("classical", classical_solution.flatten())
    print("quantum", quantum_solution.flatten())
    if plot:
        plot_csol_vs_qsol(classical_solution=classical_solution, quantum_solution=quantum_solution, title=name)

    print(f"Finished classiq run in {run_time}s.")

    if np.linalg.norm(classical_solution - quantum_solution) / np.linalg.norm(classical_solution) > 0.2:
        raise RuntimeError("The HHL solution is too far from the classical one, please verify your algorithm.")
