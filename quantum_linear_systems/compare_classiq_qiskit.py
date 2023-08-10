"""Compare the Classiq and Qiskit implementations on different use-cases and plot the results."""
import matplotlib.pyplot as plt

from quantum_linear_systems.toymodels import Qiskit4QubitExample, ClassiqDemoExample, VolterraProblem
from quantum_linear_systems.hhl_qiskit_implementation import qiskit_hhl
from quantum_linear_systems.hhl_classiq_implementation import classiq_hhl
from quantum_linear_systems.utils import (plot_compare_csol_vs_qsol,
                                          plot_depth_runtime_distance_vs_problem,
                                          relative_distance_quantum_classical_solution)

# define example (e.g. parse as arg? or do whole list)
if __name__ == "__main__":
    toymodels = [ClassiqDemoExample(), Qiskit4QubitExample(), VolterraProblem(problem_size=2),
                 VolterraProblem(problem_size=4)]

    qsols_classiq, csols_classiq, run_times_classiq, depths_classiq, widths_classiq = [], [], [], [], []
    qsols_qiskit, csols_qiskit, run_times_qiskit, depths_qiskit, widths_qiskit = [], [], [], [], []
    for model in toymodels:
        print(f"Solving {model.name}")
        # run example with classiq and qiskit (make sure they don't plot or open browser for speed)
        q_qsol, q_csol, q_depth, q_width, q_run_time = qiskit_hhl(model=model, show_circuit=False)
        c_qsol, c_csol, c_depth, c_width, c_run_time = classiq_hhl(model=model, show_circuit=False)
        assert q_width == c_width
        qsols_classiq.append(c_qsol)
        qsols_qiskit.append(q_qsol)
        csols_classiq.append(c_csol)
        csols_qiskit.append(q_csol)

        run_times_classiq.append(c_run_time)
        run_times_qiskit.append(q_run_time)

        depths_classiq.append(c_depth)
        depths_qiskit.append(q_depth)

        widths_classiq.append(c_width)
        widths_qiskit.append(q_width)

    N_PROBLEMS = len(toymodels)

    # calculate relative distance of qsol and csol
    rel_dis_classiq, rel_dis_qiskit = [], []
    for i in range(N_PROBLEMS):
        rel_dis_classiq.append(relative_distance_quantum_classical_solution(qsols_classiq[i], csols_classiq[i]))
        rel_dis_qiskit.append(relative_distance_quantum_classical_solution(qsols_qiskit[i], csols_qiskit[i]))

    # Create subplots for each problem
    fig, axs = plt.subplots(N_PROBLEMS, 2, figsize=(12, 6 * N_PROBLEMS))

    for i in range(N_PROBLEMS):
        plot_compare_csol_vs_qsol(classical_solution=csols_classiq[i], quantum_solution_classiq=qsols_classiq[i],
                                  quantum_solution_qiskit=qsols_qiskit[i], title=f"{toymodels[i].name}", ax=axs[i, 0])
    plot_depth_runtime_distance_vs_problem(depth_classiq=depths_classiq, depth_qiskit=depths_qiskit,
                                           runtime_classiq=run_times_classiq, runtime_qiskit=run_times_qiskit,
                                           distance_classiq=rel_dis_classiq, distance_qiskit=rel_dis_qiskit,
                                           problems=toymodels, axs=axs[:, 1])

    plt.tight_layout()
    plt.show()
