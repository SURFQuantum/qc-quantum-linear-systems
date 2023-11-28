import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict
from typing import Tuple

import boto3
from braket.aws import AwsQuantumJob
from braket.devices import Devices
from braket.jobs import OutputDataConfig
from braket.jobs.hybrid_job import hybrid_job
from braket.tracking import Tracker
from qiskit import QuantumCircuit
from qiskit.primitives import BackendEstimator
from qiskit.providers import ProviderV1
from qiskit.result import Result
from qiskit.visualization import plot_histogram
from qiskit_braket_provider import AWSBraketProvider
from qiskit_braket_provider import BraketLocalBackend

from quantum_linear_systems.implementations.vqls_qiskit_implementation import (
    solve_vqls_qiskit,
)
from quantum_linear_systems.plotting import print_results
from quantum_linear_systems.toymodels import ClassiqDemoExample


def run_local_aws(circuit: QuantumCircuit, shots: int = 1000) -> Result:
    """Run circuit on local AWS BraKet backend."""
    local_simulator = BraketLocalBackend()
    task = local_simulator.run(circuit, shots=shots)
    plot_histogram(task.result().get_counts())
    return task.result()


def run_real_device_aws(
    circuit: QuantumCircuit, device_name: str, shots: int = 100
) -> Result:
    """Run circuit on real AWS BraKet device."""
    provider: ProviderV1 = AWSBraketProvider()
    # select device by name
    device = None
    if device_name == "ionq":
        device = "arn:aws:braket:::device/qpu/ionq/ionQdevice"
    elif device_name == "rigetti":
        device = "arn:aws:braket:::device/qpu/rigetti/Aspen-M-1"
    elif device_name == "oqc":
        device = "arn:aws:braket:::device/qpu/oqc/Lucy"

    if device is None:
        raise ValueError(f"{device_name} not in the list of known device ARNs.")

    backend = provider.get_backend(device)
    aws_quantum_job = backend.run(circuit, shots=shots)

    check_job_status(aws_quantum_job=aws_quantum_job)
    job_result = aws_quantum_job.result()
    plot_histogram(job_result.get_counts())


def check_job_status(
    aws_quantum_job: AwsQuantumJob, seconds_interval: int = 10
) -> None:
    """Check job status every `seconds_interval` seconds until the quantum job is done
    or failed."""
    while True:
        state = aws_quantum_job.state()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if state == "COMPLETED":
            print(
                f"{current_time} - Your quantum job {aws_quantum_job.arn} has completed successfully."
            )
            # Retrieve and print job result
            result = aws_quantum_job.result()
            print("Job result:", result)
            # Print measurement probabilities if available
            if "measurementProbabilities" in result:
                print("Measurement Probabilities:", result["measurementProbabilities"])
            break
        elif state == "FAILED":
            print(f"{current_time} - Your quantum job {aws_quantum_job.arn} failed.")
            # Retrieve and print job metadata for debugging
            metadata = aws_quantum_job.metadata()
            print("Job failed with metadata:", metadata)
            # Display logs for debugging
            try:
                aws_quantum_job.logs(wait=False)
            except Exception as e:
                print("An error occurred while retrieving logs:", e)
            break
        else:
            print(
                f"{current_time} - Current status of your quantum job {aws_quantum_job.arn} is: {state}"
            )
            if state == "QUEUED":
                queue_info = aws_quantum_job.queue_position()
                if queue_info and queue_info.queue_position:
                    print(
                        f"{current_time} - Your position in the queue is {queue_info.queue_position}"
                    )
            time.sleep(seconds_interval)


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
        else Devices.Amazon.SV1
    )

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
        # define hybrid job
        model = ClassiqDemoExample()
        # model = HEPTrackReconstruction(num_detectors=5, num_particles=5)
        # runtimes(250): 3,3 =150s; 4,3=153s; 4,4=677s ;5,4=654s (c.25) ; 5,5=3492s (c0.34)
        # Note: neither memory nor cpu usage significant at these sizes
        # Note: after 250 iterations the cost is not low enough, would it make more sense to define different stop criteria

        # define estimator
        backend = AWSBraketProvider().get_backend(name=device_arn)
        estimator = BackendEstimator(backend=backend, skip_transpilation=False)

        qsol, _, depth, width, run_time = solve_vqls_qiskit(
            matrix_a=model.matrix_a,
            vector_b=model.vector_b,
            show_circuit=True,
            estimator=estimator,
        )

        print_results(
            quantum_solution=qsol,
            classical_solution=model.classical_solution,
            run_time=run_time,
            name=model.name,
            plot=True,
        )

    with Tracker() as tracker:
        # submit the job
        job: AwsQuantumJob = execute_hybrid_job()

        check_job_status(aws_quantum_job=job, seconds_interval=10)

        # Check the final status
        print(f"Job {job.arn} finished with status {job.state()}.")

        # Retrieve results if job is completed
        if job.state() == "COMPLETED":
            result = job.result()
            print("Job result:", result)
            # display the results
            print(result.measurement_counts)
    print(tracker.simulator_tasks_cost())
