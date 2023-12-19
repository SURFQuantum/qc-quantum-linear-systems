"""This module implements a naive setup of a VQLS hybrid task where the quantum part is
saved as a qasm file and then submitted to the quantum hardware.

The results are then given to the classical part of the algorithm. This is looped over
in a for loop.
"""
import time
from typing import Any
from typing import Callable
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms.minimum_eigen_solvers.vqe import _validate_bounds
from qiskit.algorithms.minimum_eigen_solvers.vqe import _validate_initial_point
from qiskit.algorithms.optimizers import Minimizer
from qiskit.algorithms.optimizers import Optimizer
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit.primitives import Estimator
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.optimizers import SLSQP
from vqls_prototype import VQLS

from quantum_linear_systems.implementations.vqls_qiskit_implementation import (
    postprocess_solution,
)
from quantum_linear_systems.plotting import print_results
from quantum_linear_systems.toymodels import ClassiqDemoExample
from quantum_linear_systems.utils import circuit_to_qasm3


def naive_hybrid_solve_vqls(
    matrix_a: np.ndarray,
    vector_b: np.ndarray,
    ansatz: QuantumCircuit = None,
    estimator: Estimator = Estimator(),
    optimizer_name: str = "cobyla",
    optimizer_max_iter: int = 250,
    show_circuit: bool = False,
) -> Tuple[np.ndarray, str, int, int, float]:
    start_time = time.time()
    np.set_printoptions(precision=3, suppress=True)

    if ansatz is None:
        ansatz = RealAmplitudes(
            num_qubits=int(np.log2(matrix_a.shape[0])),
            entanglement="full",
            reps=3,
            insert_barriers=False,
        )
    if optimizer_name.lower() == "cobyla":
        optimizer = COBYLA(maxiter=1, disp=True)
    elif optimizer_name.lower() == "slsqp":
        optimizer = SLSQP(maxiter=1, disp=True)
    else:
        raise ValueError(f"Invalid optimizer_name: {optimizer_name}")

    if vector_b.ndim == 2:
        vector_b = vector_b.flatten()

    vqls = VQLS(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=Sampler(),
        options={
            "use_overlap_test": False,
            "use_local_cost_function": False,
            "verbose": True,
        },
    )
    print("--------------")
    # Note: copied from vqls._solve()

    # compute the circuits needed for the hadamard tests
    hdmr_tests_norm, hdmr_tests_overlap = vqls.construct_circuit(matrix_a, vector_b)
    print("len constructed circ ", len(hdmr_tests_overlap) + len(hdmr_tests_norm))
    # compute he coefficient matrix
    coefficient_matrix = vqls.get_coefficient_matrix(
        np.array([mat_i.coeff for mat_i in vqls.matrix_circuits])
    )
    print("coeff matrix: ", coefficient_matrix)

    # set an expectation for this algorithm run (will be reset to None at the end)
    initial_point = _validate_initial_point(vqls.initial_point, vqls.ansatz)
    bounds = _validate_bounds(vqls.ansatz)
    print(initial_point)

    # Convert the gradient operator into a callable function that is compatible with the
    # optimization routine.
    gradient = vqls._gradient
    vqls._eval_count = 0

    # get the cost evaluation function
    cost_evaluation = vqls.get_cost_evaluation_function(
        hdmr_tests_norm, hdmr_tests_overlap, coefficient_matrix
    )
    print("Cost_evaluation: ", cost_evaluation)
    print("-----")
    res = vqls.solve(matrix_a, vector_b)

    # Step 1: determine initial set of parameters
    estimator_parameters = estimator.parameters

    initial_parameters: List[float] = estimator_parameters
    if initial_parameters is None:
        print("Step 1: Not implemented yet.")
    # Step 2: get circuits from estimator and assign initial parameters
    estimator_circuits = estimator.circuits

    print("Estimator circuits", estimator_circuits)
    print("Estimator parameters", estimator_parameters)

    for n in range(optimizer_max_iter):
        new_params, current_cost = naive_hybrid_loop(
            circuits=estimator_circuits,
            parameter_names=estimator_parameters,
            parameter_values=initial_parameters,
            optimizer=vqls.optimizer[0],
            gradient=gradient,
            bounds=bounds,
        )
        initial_parameters = new_params
        print(f"Cost={current_cost} in iteration {n}/{optimizer_max_iter}.")

    # Step 6: create output state from final set of params
    sol_circuit = ansatz.assign_parameters(
        {
            param: value
            for param, value in zip(
                estimator_parameters[0],
                initial_parameters,
            )
        }
    )
    print("Step 6: Not implemented yet.")

    vqls_circuit = sol_circuit
    vqls_solution_vector = np.real(Statevector(res.state).data)

    quantum_solution = postprocess_solution(
        matrix_a=matrix_a, vector_b=vector_b, solution_x=vqls_solution_vector
    )

    qc_basis = vqls_circuit.decompose(reps=10)

    if show_circuit:
        print(qc_basis)

    # todo: fix, make sure this is the right circuit
    qasm_content = circuit_to_qasm3(
        circuit=vqls_circuit, filename="vqls_qiskit_circuit.qasm3"
    )

    print(
        f"Comparing depths original {vqls_circuit.depth()} vs. decomposed {qc_basis.depth()}"
    )

    return (
        quantum_solution,
        qasm_content,
        qc_basis.depth(),
        vqls_circuit.width(),
        time.time() - start_time,
    )


def naive_hybrid_loop(
    circuits: List[QuantumCircuit],
    parameter_names: List[str],
    parameter_values: List[float],
    optimizer: Union[Optimizer, Minimizer],
    gradient: Callable[[Any], Any],
    bounds: list[tuple[float, float]],
) -> Tuple[List[float], float]:
    qasm_circuits: List[str] = []
    for i, circ in enumerate(circuits):
        print(circ.name)
        # Bind parameters to specific values
        bound_circ = circ.assign_parameters(
            {
                param: value
                for param, value in zip(
                    parameter_names[i],
                    list(np.zeros(len(parameter_names[i]))),
                    # parameter_values,
                )
            }
        )
        qasm_circuits.append(bound_circ.qasm())
        print(bound_circ.draw())

    # Step 3: Execute on quantum backend
    def execute_quantum(qasm_circs: List[str]) -> None:
        """Execute the different circuits on the quantum device."""
        print("Step 3: Not implemented yet.")

    # Step 4: minimize cost function
    def cost_function(x0: List[float]) -> float:
        """VQLS cost function."""
        return x0[0] * 0.0

    print(optimizer)
    results = optimizer.minimize(
        fun=cost_function,
        x0=list(np.zeros(len(parameter_values))),
        jac=gradient,
        bounds=bounds,
    )
    print("Step 4: Not implemented yet.")

    # Step 5: get new parameters to start
    new_params = results.x
    cost = results.fun

    return new_params, cost


if __name__ == "__main__":
    N = 1

    model = ClassiqDemoExample()
    # model = HEPTrackReconstruction(num_detectors=5, num_particles=5)
    # runtimes(250): 3,3 =150s; 4,3=153s; 4,4=677s ;5,4=654s (c.25) ; 5,5=3492s (c0.34)
    # Note: neither memory nor cpu usage significant at these sizes
    # Note: after 250 iterations the cost is not low enough, would it make more sense to define different stop criteria
    qsol, _, depth, width, run_time = naive_hybrid_solve_vqls(
        matrix_a=model.matrix_a,
        vector_b=model.vector_b,
        show_circuit=True,
        optimizer_max_iter=1,
    )

    print_results(
        quantum_solution=qsol,
        classical_solution=model.classical_solution,
        run_time=run_time,
        name=model.name,
        plot=False,
    )
