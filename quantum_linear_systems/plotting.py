"""Plotting functions that can be imported by either implementation."""
from typing import List
from typing import Tuple

import matplotlib.axis
import matplotlib.pyplot as plt
import numpy as np

from quantum_linear_systems.toymodels import ToyModel


def plot_csol_vs_qsol(
    classical_solution: np.ndarray, quantum_solution: np.ndarray, title: str
) -> None:
    """Plot classical and quantum solution vectors side by side.

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


def plot_compare_csol_vs_qsol(
    classical_solution: np.ndarray,
    qsols_marker_name: List[Tuple[List[np.ndarray], str, str]],
    title: str,
    axis: matplotlib.axis.Axis = None,
) -> None:
    """Plot classical and quantum solutions side by side.

    Parameters:
        classical_solution (numpy.ndarray): Array representing the classical solution.
        qsols_marker_name: (List[Tuple[list, str, str]]): List of tuples with first element List of quantum solutions,
        second element the desired marker type and third element the corresponding label.
        title (str): Title for the plot.
        axis (matplotlib.axes._subplots.AxesSubplot): Axes to use for the plot. If None, a new subplot will be created.
    """
    if axis is None:
        _, axis = plt.subplots()

    axis.plot(classical_solution, "bs", label="classical")
    for qmn in qsols_marker_name:
        axis.plot(qmn[0], qmn[1], label=qmn[2])
    axis.legend()
    axis.set_xlabel("$i$")
    axis.set_ylabel("$x_i$")
    axis.set_ylim(0, 1)
    axis.set_title(title)
    axis.grid(True)


def plot_depth_runtime_distance_vs_problem(
    depth_runtime_distance_marker_name: List[
        Tuple[List[int], List[float], List[float], str, str]
    ],
    problems: List[ToyModel],
    axs: List[matplotlib.axis.Axis] = None,  # type: ignore[assignment]
) -> None:
    """Plot depth and runtime of two algorithms side by side for each problem index.

    Parameters:
        depth_runtime_distance_marker_name (list) : List of tuples of the form (depths, run_times, rel_distance,
        marker, name).
        problems (list): List of problem objects with a 'name' attribute.
        axs (list): List of axes to use for the plot. If None, a new subplot will be created.
    """
    problem_names = [problem.name for problem in problems]

    if axs is None or len(axs) < 3:
        _, axs = plt.subplots(3, 1, figsize=(10, 6))
    for drdmn in depth_runtime_distance_marker_name:
        axs[0].plot(problem_names, drdmn[0], drdmn[3], label=drdmn[4])
    axs[0].legend()
    axs[0].set_ylabel("Circuit Depth")
    axs[0].set_title("Circuit Depth Comparison")
    axs[0].grid(True)

    for drdmn in depth_runtime_distance_marker_name:
        axs[1].plot(problem_names, drdmn[1], drdmn[3], label=drdmn[4])
    axs[1].legend()
    axs[1].set_ylabel("Run Time [s]")
    axs[1].set_title("Run Time Comparison")
    axs[1].grid(True)
    axs[1].set_yscale("log")

    for drdmn in depth_runtime_distance_marker_name:
        axs[2].plot(problem_names, drdmn[2], drdmn[3], label=drdmn[4])
    axs[2].legend()
    axs[2].set_ylabel("Relative Distance [%]")
    axs[2].set_title("Rel. Distance of Quantum/Classical Solution")
    axs[2].grid(True)


def print_results(
    quantum_solution: np.ndarray,
    classical_solution: np.ndarray,
    run_time: float,
    name: str,
    plot: bool = True,
) -> None:
    """Print results of classical and quantum solutions and optionally plot them.

    Parameters:
        quantum_solution (numpy.ndarray): Quantum solution.
        classical_solution (numpy.ndarray): Classical solution.
        run_time (float): Time taken for the computation.
        name (str): Name of the solution.
        plot (bool, optional): Whether to generate and display a plot. Default is True.
    """
    classical_solution /= np.linalg.norm(classical_solution)
    quantum_solution /= np.linalg.norm(quantum_solution)
    print("classical", classical_solution.flatten())
    print("quantum", quantum_solution.flatten())
    if plot:
        plot_csol_vs_qsol(
            classical_solution=classical_solution,
            quantum_solution=quantum_solution,
            title=name,
        )

    print(f"Finished run in {run_time}s.")

    if (
        np.linalg.norm(classical_solution - quantum_solution)
        / np.linalg.norm(classical_solution)
        > 0.2
    ):
        raise RuntimeError(
            "The HHL solution is too far from the classical one, please verify your algorithm."
        )
