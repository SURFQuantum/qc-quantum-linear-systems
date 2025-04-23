"""HHL implementation using Classiq."""

import time
from typing import Tuple, Optional


import numpy as np
import scipy
from classiq import QuantumProgram
from classiq import (
    execute,
    qfunc,
    CInt,
    CArray,
    QNum,
    Output,
    QBit,
    allocate,
    CReal,
    QCallable,
    QArray,
    prepare_amplitudes,
    within_apply,
    unitary,
    create_model,
    set_preferences,
    CustomHardwareSettings,
    Preferences,
    synthesize,
)
from classiq.open_library import qpe
from classiq.open_library import allocate_num
from classiq.execution import (
    ClassiqBackendPreferences,
    ExecutionPreferences,
)
from classiq.synthesis import show
from classiq.synthesis import set_execution_preferences
from classiq.interface.backend.backend_preferences import ClassiqSimulatorBackendNames

from quantum_linear_systems.implementations.vqls_qiskit_implementation import (
    postprocess_solution,
)
from quantum_linear_systems.plotting import print_results
from quantum_linear_systems.toymodels import ClassiqDemoExample


@qfunc
def simple_eig_inv(phase: QNum, indicator: Output[QBit]) -> None:
    allocate(1, indicator)
    indicator *= (1 / 2**phase.size) / phase


@qfunc
def classiq_hhl(
    precision: CInt,
    b: CArray[CReal],
    unitary: QCallable[QArray[QBit]],
    res: Output[QArray[QBit]],
    phase: Output[QNum],
    indicator: Output[QBit],
) -> None:
    prepare_amplitudes(b, 0.0, res)
    allocate_num(precision, False, precision, phase)
    within_apply(
        lambda: qpe(unitary=lambda: unitary(res), phase=phase),
        lambda: simple_eig_inv(phase=phase, indicator=indicator),
    )


def solve_hhl_classiq(
    matrix_a: np.ndarray,
    vector_b: np.ndarray,
    qpe_register_size: Optional[int] = None,
    show_circuit: bool = False,
) -> Tuple[np.ndarray, str, int, int, float]:
    """This function models, synthesizes, executes an HHL example and returns the depth,
    cx-counts and fidelity.

    Classiq HHL implementation based on
    https://docs.classiq.io/latest/explore/tutorials/technology_demonstrations/hhl/hhl_example/
    .
    """

    start_time = time.time()

    # SP params
    b_normalized = vector_b.tolist()

    solution_register_size = int(np.log2(len(vector_b)))
    if qpe_register_size is None:
        # calculate size of qpe_register from matrix
        kappa = np.linalg.cond(matrix_a)  # condition number of matrix
        neg_vals = True  # whether matrix has negative eigenvalues
        qpe_register_size = (
            max(solution_register_size + 1, int(np.ceil(np.log2(kappa + 1)))) + neg_vals
        )
    print(
        f"Size of solution register is {solution_register_size} , QPE registers is {qpe_register_size}."
    )
    precision = qpe_register_size
    # sp_upper = 0.00  # precision of the State Preparation
    num_qubits = int(np.log2(len(vector_b)))
    # exact unitary
    exact_unitary = scipy.linalg.expm(1j * 2 * np.pi * matrix_a)
    unitary_mat = exact_unitary.tolist()

    @qfunc
    def main(res: Output[QNum], phase: Output[QNum], indicator: Output[QBit]) -> None:
        classiq_hhl(
            precision=precision,
            b=b_normalized,
            unitary=lambda target: unitary(elements=unitary_mat, target=target),
            res=res,
            phase=phase,
            indicator=indicator,
        )

    qmod_hhl = create_model(main)
    backend_preferences = ClassiqBackendPreferences(
        backend_name=ClassiqSimulatorBackendNames.SIMULATOR_STATEVECTOR
    )
    qmod_hhl = set_preferences(
        qmod_hhl,
        custom_hardware_settings=CustomHardwareSettings(basis_gates=["cx", "u"]),
        transpilation_option="auto optimize",
    )
    qmod_hhl = set_execution_preferences(
        qmod_hhl,
        execution_preferences=ExecutionPreferences(
            num_shots=1, backend_preferences=backend_preferences
        ),
    )

    # Synthesize
    synth_prefs = Preferences(output_format=["qasm"], pretty_qasm=True, qasm3=True)
    qprog_hhl = synthesize(qmod_hhl, preferences=synth_prefs)
    if show_circuit:
        show(qprog_hhl)
    qasm_content = QuantumProgram.from_qprog(qprog_hhl).qasm

    circuit_hhl = QuantumProgram.from_qprog(qprog_hhl)
    circuit_width = (
        circuit_hhl.data.width
    )  # total number of qubits of the whole circuit
    circuit_depth = circuit_hhl.transpiled_circuit.depth
    cx_counts = circuit_hhl.transpiled_circuit.count_ops["cx"]

    # Execute
    result = execute(qprog_hhl).result_value()

    # Post-process
    # target_pos = result.physical_qubits_map["indicator"][0]  # position of control qubit
    # sol_pos = list(result.physical_qubits_map["res"])  # position of solution
    # phase_pos = list(
    #     result.physical_qubits_map["phase"]
    # )  # position of the “phase” register, and flips for endianness as we will use the indices to read directly from the string
    qsol = [
        np.round(parsed_state.amplitude / (1 / 2**precision), 5)
        for solution in range(2**num_qubits)
        for parsed_state in result.parsed_state_vector
        if parsed_state["indicator"] == 1.0
        and parsed_state["res"] == solution
        and parsed_state["phase"]
        == 0.0  # this takes the entries where the “phase” register is at state zero
    ]
    quantum_solution = postprocess_solution(
        matrix_a=matrix_a, vector_b=vector_b, solution_x=np.array(qsol)
    )

    # return total_q, depth, cx_counts
    print(f"Total number of operations in classiq circuit: {cx_counts}")
    return (
        quantum_solution,
        qasm_content,
        circuit_depth,
        circuit_width,
        time.time() - start_time,
    )


if __name__ == "__main__":
    # input params
    N: int = 2

    model = ClassiqDemoExample()

    qsol, _, depth, width, run_time = solve_hhl_classiq(
        matrix_a=model.matrix_a, vector_b=model.vector_b, show_circuit=True
    )

    print_results(
        quantum_solution=qsol,
        classical_solution=model.classical_solution,
        run_time=run_time,
        name=model.name,
        plot=True,
    )
