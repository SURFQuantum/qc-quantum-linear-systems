import argparse
import json
import os
import time
from typing import Dict
from typing import Tuple

import boto3
import numpy as np
from braket.aws import AwsQuantumJob
from braket.jobs import OutputDataConfig
from braket.jobs.hybrid_job import hybrid_job
from braket.tracking import Tracker
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import BackendEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit_braket_provider import AWSBraketProvider


def get_tags() -> Dict[str, str]:
    with open("/etc/src_quantum.json", "r") as fp:
        config = json.load(fp)
    return {
        "workspace_id": config["workspace_id"],
        "subscription": config["subscription"],
    }


def aws_s3_folder(folder_name: str) -> Tuple[str, str]:
    with open("/etc/src_quantum.json", "r") as fp:
        config = json.load(fp)
    bucket = f"amazon-braket-{config['workspace_id']}"
    return (
        bucket,
        folder_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AWS run script for a hybrid job on a simulator or a real quantum device."
    )
    parser.add_argument(
        "--real", action="store_true", help="Use a real device instead of a simulator"
    )
    parser.add_argument(
        "--local", action="store_true", help="Run the hybrid job locally"
    )
    args = parser.parse_args()

    # SURF-ResearchCloud setup
    my_prefix = "quantum_linear_systems"
    s3_folder = aws_s3_folder(my_prefix)
    output_data_config = OutputDataConfig(
        s3Path=f"s3://{s3_folder[0]}/{s3_folder[1]}/output"
    )
    # set region
    os.environ["AWS_DEFAULT_REGION"] = "eu-west-2"
    # get account
    aws_account_id = boto3.client("sts").get_caller_identity()["Account"]
    # set device
    device_arn = (
        "arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy"
        if args.real
        else "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
    )
    device_name = "oqc/Lucy" if args.real else "SV1"

    # Define the role ARN for executing the hybrid job (replace with your actual role ARN)
    surf_role_arn = (
        "arn:aws:iam::815925483357:role/src-workspace-AmazonBraketJobsExecutionRole"
    )

    @hybrid_job(
        device=device_arn,
        role_arn=surf_role_arn,
        output_data_config=output_data_config,
        dependencies="aws_requirements.txt",
        local=args.local,
        tags=get_tags(),
    )  # choose priority device
    def execute_hybrid_job() -> None:
        # define estimator
        backend = AWSBraketProvider().get_backend(name=device_name)
        estimator = BackendEstimator(backend=backend, skip_transpilation=False)

        # Create a simple circuit
        theta = Parameter("θ")
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.rz(theta, 0)

        observable = SparsePauliOp(["II", "XZ"])

        # Define a cost function
        def cost_function(param: np.ndarray) -> float:
            bound_circuit = circuit.bind_parameters({theta: param[0]})
            job = estimator.run([bound_circuit], [observable])
            cost = 1 - job.result().values[0]  # 1 - expectation value
            return float(cost)

        # Use a classical optimizer
        optimizer = COBYLA(maxiter=100)
        initial_point = [0.0]
        result = optimizer.minimize(fun=cost_function, x0=initial_point)
        optimal_point = result.x
        value = result.fun

        print("Optimal point:", optimal_point)
        print("Optimal value:", value)

    with Tracker() as tracker:
        # submit the job
        job: AwsQuantumJob = execute_hybrid_job()

        while True:
            state = job.state()
            if state in ["COMPLETED", "FAILED"]:
                break
            else:
                print(f"{job} submitted, not done yet.")
                time.sleep(10)

    print(tracker.simulator_tasks_cost())
