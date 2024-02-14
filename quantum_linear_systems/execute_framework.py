import argparse

import numpy as np
import yaml  # type: ignore[import-untyped]

from quantum_linear_systems.quantum_linear_solver import QuantumLinearSolver
from quantum_linear_systems.toymodels import ClassiqDemoExample


def parse_arguments() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(description="Quantum Linear Systems Framework")
    arg_parser.add_argument(
        "-m",
        "--matrix_csv",
        help="CSV file containing the matrix A.",
        required=False,
        type=str,
    )
    arg_parser.add_argument(
        "-v",
        "--vector_csv",
        help="CSV file containing the vector b.",
        required=False,
        type=str,
    )
    arg_parser.add_argument(
        "-i",
        "--implementation",
        help="Implementation to solve the problem Ax=b.",
        required=True,
        type=str,
    )
    arg_parser.add_argument(
        "-iargs",
        "--implementation_args",
        type=str,
        help="Path to a YAML file containing a specific parameters to be passed to the implementation.",
        required=False,
    )

    args = arg_parser.parse_args()

    if args.matrix_csv:
        args.matrix_a = np.loadtxt(args.matrix_csv, delimiter=",")
    else:
        print("No matrix provided. Falling back to ToyModel.")
        toymodel = ClassiqDemoExample()
        args.matrix_a = toymodel.matrix_a
    if args.vector_csv:
        args.vector_b = np.loadtxt(args.vector_csv, delimiter=",")
    else:
        print("No vector provided. Falling back to ToyModel.")
        toymodel = ClassiqDemoExample()
        args.vector_b = toymodel.vector_b
    if args.implementation not in ["hhl_qiskit", "vqls_qiksit", "hhl_classiq"]:
        raise ValueError(f"Unknown implementation {args.implementation}.")
    if args.implementation_args:
        with open(args.implementation_args, "r") as file:
            try:
                args.implementation_args = yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise ValueError(f"Error loading YAML argument file: {e}")
    else:
        args.implementation_args = {}

    return args


if __name__ == "__main__":
    parsed_args = parse_arguments()

    qls = QuantumLinearSolver()

    # perform checks (add more meaningful checks here, also implementation specific checks)
    qls.check_matrix_square_hermitian(parsed_args.matrix_a)

    # solve
    qls.solve(
        matrix_a=parsed_args.matrix_a,
        vector_b=parsed_args.vector_b,
        method=parsed_args.implementation,
        file_basename="default_run",
        **parsed_args.implementation_args,
    )

    # todo: backend execution
    # this is currently still hardcoded into the implementations (with the exception of `hhl_qiskit`)
    # we do have access to the qasm circuits through `qls.qasm_circuit`.
    # However, especially with hybrid algorithms such as VQLS, we need to think about how to actually execute them on different backends

    print(
        f"Solution: {qls.solution}\nCircuit depth: {qls.circuit_depth}\nCircuit width: {qls.circuit_width}\nRuntime: {qls.run_time}"
    )
