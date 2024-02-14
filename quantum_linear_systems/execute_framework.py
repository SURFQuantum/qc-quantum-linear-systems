import argparse

import numpy as np

from quantum_linear_systems.quantum_linear_solver import QuantumLinearSolver
from quantum_linear_systems.toymodels import ClassiqDemoExample

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
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

    args = arg_parser.parse_args()

    if args.matrix_csv is None:
        print("No matrix provided. Falling back to ToyModel.")
        toymodel = ClassiqDemoExample()
        matrix_a = toymodel.matrix_a
    else:
        matrix_a = np.loadtxt(args.matrix_csv, delimiter=",")
    if args.vector_csv is None:
        print("No vector provided. Falling back to ToyModel.")
        toymodel = ClassiqDemoExample()
        vector_b = toymodel.vector_b
    else:
        vector_b = np.loadtxt(args.vector_csv, delimiter=",")

    if args.implementation not in ["hhl_qiskit", "vqls_qiksit", "hhl_classiq"]:
        raise ValueError(f"Unknown implementation {args.implementation}.")

    qls = QuantumLinearSolver()

    # perform checks (add more meaningful checks here)
    qls.check_matrix_square_hermitian(matrix_a)

    # solve
    qls.solve(
        matrix_a=matrix_a,
        vector_b=vector_b,
        method=args.implementation,
        file_basename="default_run",
    )

    print(
        f"Solution: {qls.solution}\nCircuit depth: {qls.circuit_depth}\nCircuit width: {qls.circuit_width}\nRuntime: {qls.run_time}"
    )
