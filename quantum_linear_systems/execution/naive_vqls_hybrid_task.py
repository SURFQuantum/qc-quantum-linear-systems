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
from qiskit import execute
from qiskit import QuantumCircuit
from qiskit.algorithms.minimum_eigen_solvers.vqe import _validate_bounds
from qiskit.algorithms.minimum_eigen_solvers.vqe import _validate_initial_point
from qiskit.algorithms.optimizers import Minimizer
from qiskit.algorithms.optimizers import Optimizer
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit.primitives import Estimator
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit.result import QuasiDistribution
from qiskit_aer import QasmSimulator
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
    # Note: copied from vqls._solve()
    # compute the circuits needed for the hadamard tests
    hdmr_tests_norm, hdmr_tests_overlap = vqls.construct_circuit(matrix_a, vector_b)
    hdmr_norm_circuits = [c for h in hdmr_tests_norm for c in h.circuits]
    hdmr_overlap_circuits = [c for h in hdmr_tests_overlap for c in h.circuits]
    print("obserables", vqls.estimator.observables)

    # compute the coefficient matrix
    coefficient_matrix = vqls.get_coefficient_matrix(
        np.array([mat_i.coeff for mat_i in vqls.matrix_circuits])
    )

    # set an expectation for this algorithm run (will be reset to None at the end)
    # initial_point = _validate_initial_point(vqls.initial_point, vqls.ansatz)
    bounds = _validate_bounds(vqls.ansatz)

    # Convert the gradient operator into a callable function that is compatible with the
    # optimization routine.
    gradient = vqls._gradient
    # vqls._eval_count = 0

    # get the cost evaluation function
    # cost_evaluation = vqls.get_cost_evaluation_function(
    #     hdmr_tests_norm, hdmr_tests_overlap, coefficient_matrix
    # )

    # Step 1: determine initial set of parameters
    parameter_names = ansatz.parameters

    initial_parameters: List[float] = _validate_initial_point(
        vqls.initial_point, vqls.ansatz
    )
    # Step 2: get circuits from estimator and assign initial parameters

    for n in range(optimizer_max_iter):
        new_params, current_cost = naive_hybrid_loop(
            norm_circuits=hdmr_norm_circuits,
            overlap_circuits=hdmr_overlap_circuits,
            parameter_names=parameter_names,
            parameter_values=initial_parameters,
            optimizer=vqls.optimizer,
            gradient=gradient,
            bounds=bounds,
            coeff_matrix=coefficient_matrix,
            vqls_instance=vqls,
        )
        initial_parameters = new_params
        print(f"Cost={current_cost} in iteration {n}/{optimizer_max_iter}.")

    # Step 6: create output state from final set of params
    vqls_circuit = ansatz.assign_parameters(
        {
            param: value
            for param, value in zip(
                parameter_names,
                initial_parameters,
            )
        }
    )
    raise NotImplementedError("Step 6: Not implemented yet.")

    vqls_solution_vector = np.real(Statevector(vqls_circuit).data)

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
    norm_circuits: List[QuantumCircuit],
    overlap_circuits: List[QuantumCircuit],
    parameter_names: List[str],
    parameter_values: List[float],
    optimizer: Union[Optimizer, Minimizer],
    gradient: Callable[[Any], Any],
    bounds: list[tuple[float, float]],
    coeff_matrix: np.ndarray,  # also for evaluation of cost function
    vqls_instance: VQLS,  # so we don't have to copy everything for the cost function
) -> Tuple[List[float], float]:
    norm_qasm = [
        circ.assign_parameters(
            {param: value for param, value in zip(parameter_names, parameter_values)}
        ).qasm()
        for i, circ in enumerate(norm_circuits)
    ]

    overlap_qasm = [
        circ.assign_parameters(
            {param: value for param, value in zip(parameter_names, parameter_values)}
        ).qasm()
        for i, circ in enumerate(overlap_circuits)
    ]
    qasm_circuits = (norm_qasm, overlap_qasm)

    # Step 3: Execute on quantum backend
    def execute_quantum(
        qasm_circs: Tuple[List[str], List[str]],
    ) -> Tuple[List[float], List[float]]:
        """Execute the different circuits on the quantum device."""
        # todo: for QuantumInspire this needs to be replaced with what they do, here a qiskit implementation:
        backend = QasmSimulator()
        norm_qasm, overlap_qasm = qasm_circs
        # todo: here we need to figure out how we extract the desired quantity from the results
        norm_results = []
        for qcirc in norm_qasm:
            qc = QuantumCircuit.from_qasm_str(qcirc)
            # Add measurements to the circuit if not already present
            qc.measure_all()
            job = execute(qc, backend)
            result = job.result().get_counts()
            quasi_dist = QuasiDistribution(result)
            norm_results.append(quasi_dist)
        overlap_results = []
        for qcirc in overlap_qasm:
            qc = QuantumCircuit.from_qasm_str(qcirc)
            # Add measurements to the circuit if not already present
            qc.measure_all()
            job = execute(qc, backend)
            result = job.result().get_counts()
            quasi_dist = QuasiDistribution(result)
            overlap_results.append(quasi_dist)
        print(norm_results)
        norm_results = test_post_processing(
            norm_results,
            num_qubits=vqls_instance._get_local_circuits()[0].num_qubits,
            post_process_coeffs=vqls_instance._get_local_circuits()[
                0
            ].post_process_coeffs,
        )
        norm_results = np.array([1.0 - 2.0 * val for val in norm_results]).astype(
            "complex128"
        )
        norm_results *= np.array([1.0, 1.0j])
        print(norm_results)
        exit()
        return norm_results, overlap_results

    norm_res, overlap_res = execute_quantum(qasm_circs=qasm_circuits)

    # Step 4: minimize cost function
    def cost_function(x0: List[float]) -> float:
        """VQLS cost function."""
        # Note since we already evaluated the circuits the input params are not used here
        # Note: copied from vqls
        cost_value = vqls_instance._assemble_cost_function(
            hdmr_values_norm=norm_res,
            hdmr_values_overlap=overlap_res,
            coefficient_matrix=coeff_matrix,
        )
        return float(cost_value)

    results = optimizer.minimize(
        fun=cost_function,
        x0=parameter_values,
        jac=gradient,
        bounds=bounds,
    )

    # Step 5: get new parameters to start
    new_params = results.x
    cost = results.fun

    return new_params, cost


def test_post_processing(  # type: ignore
    sampler_result, num_qubits: int, post_process_coeffs
) -> np.ndarray:
    """Post process the sampled values of the circuits.

    Args:
        sampler_result (results): Result of the sampler

    Returns:
        List: value of the overlap hadammard test
    """

    # quasi_dist = sampler_result.quasi_dists
    quasi_dist = sampler_result
    output = []

    for qdist in quasi_dist:
        # add missing keys
        val = np.array([qdist[k] if k in qdist else 0 for k in range(2**num_qubits)])

        value_0, value_1 = val[0::2], val[1::2]
        proba_0 = (value_0 * post_process_coeffs).sum()
        proba_1 = (value_1 * post_process_coeffs).sum()

        output.append(proba_0 - proba_1)

    return np.array(output).astype("complex128")


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
