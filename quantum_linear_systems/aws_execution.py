import time
from datetime import datetime

from qiskit import QuantumCircuit
from qiskit.providers import JobStatus
from qiskit.providers import ProviderV1
from qiskit.result import Result
from qiskit.visualization import plot_histogram
from qiskit_braket_provider import AWSBraketProvider
from qiskit_braket_provider import BraketLocalBackend
from qiskit_braket_provider.providers.braket_job import AmazonBraketTask


def run_local_aws(circuit: QuantumCircuit, shots: int = 1000) -> Result:
    """Run circuit on local AWS BraKet backend."""
    local_simulator = BraketLocalBackend()
    task = local_simulator.run(circuit, shots=shots)
    plot_histogram(task.result().get_counts())
    return task.result()


def run_real_device_aws(circuit: QuantumCircuit, device_name: str, shots=100) -> Result:
    """Run circuit on real AWS BraKet device."""
    provider: ProviderV1 = AWSBraketProvider()
    # select device by name
    if device_name == "ionq":
        device = provider.get_backend("IonQ Device")
    elif device_name == "rigetti":
        device = provider.get_backend("Aspen-M-1")
    elif device_name == "oqc":
        device = provider.get_backend("Lucy")
    else:
        return ValueError(f"{device_name} not in the list of known device names.")

    task = device.run(circuit, shots=shots)

    provider.get_backend("Lucy").run()

    retrieved_job: AmazonBraketTask = device.retrieve_job(job_id=task.job_id())

    check_task_status(braket_task=retrieved_job)
    result = retrieved_job.result()
    plot_histogram(result.get_counts())


def check_task_status(
    braket_task: AmazonBraketTask, seconds_interval: int = 10
) -> None:
    """Check task status every `second_interval` seconds until the quantum task is done."""
    while True:
        status = braket_task.status()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if status == JobStatus.DONE:
            print(f"{current_time} - Your quantum task {braket_task.task_id} is done!")
            break  # Exit the loop if the job is done
        else:
            print(
                f"{current_time} - Current status of your quantum task {braket_task.task_id} is: {status}"
            )
            if status == JobStatus.QUEUED:
                print(
                    f"{current_time} - Your position in the queue is {braket_task.queue_position()}"
                )
            time.sleep(seconds_interval)


if __name__ == "__main__":
    dev = "ionq"
    qcirc = QuantumCircuit(3)
    qcirc.draw("mpl")
