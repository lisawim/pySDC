import numpy as np
import datetime
import dill
import glob
import os
import shutil
import subprocess

from pySDC.projects.DAE import (
    getColor, getColorQI, getMarker, getMarkerQI, getLabel, get_linestyle, get_linestyle_QI, Plotter
)

from pySDC.projects.DAE.run.error import get_problem_cases, plot_result


def base_args(log_time, problem_name):
    return {
        "problem_name": problem_name,
        "log_time": log_time,
    }

def extract_eps_from_filename(fname):
    if "eps" in fname:
        eps_token = fname.split("eps")[-1].split("_")[0].replace("p", ".")
        try:
            return float(eps_token)
        except ValueError:
            return 0.0
    return 0.0

def plot_work_error(case, error_type, log_time, num_nodes, problem_name, QI_list, separate_errors, solver_type):
    result_dir = os.path.join("data", "latest")
    if not os.path.isdir(result_dir):
        raise FileNotFoundError("No current result path found. Please run a simulation first.")

    problems = get_problem_cases(k=case, problem_name=problem_name)
    single_sweeper = len(problems) == 1

    results = []
    results_diff = []
    results_alg = []

    for fname in sorted(glob.glob(os.path.join(result_dir, "result_*.pkl"))):
        with open(fname, "rb") as f:
            data = dill.load(f)

        base = os.path.splitext(os.path.basename(fname))[0]  # z.B. 'result_MIN-SR-S_constrainedDAE_eps0p0'
        _, QI, *rest = base.split("_")                       # ['result', 'MIN-SR-S', 'constrainedDAE', 'eps0p0']

        eps_token = rest[-1]
        problem_type = "_".join(rest[:-1])
        eps = float(eps_token.replace("eps", "").replace("p", "."))

        eps = 0.0
        try:
            q = QI_list.index(QI)
        except ValueError:
            q = 0
        
        if single_sweeper or len(QI_list) == 2:
            q = 0

        eps = extract_eps_from_filename(fname)
        eps_values = problems.get(problem_type, [0.0])
        try:
            i = eps_values.index(eps)
        except ValueError:
            i = 0

        if isinstance(data, tuple):
            err = data[0]
            err_diff = data[1] if len(data) > 1 else None
            err_alg = data[2] if len(data) > 2 else None
            work = data[3] if len(data) > 3 else None
        else:
            err = data
            work = list(range(len(err)))
            err_diff = err_alg = None

        results.append((q, work, err, problem_type, i, QI, eps))
        if err_diff is not None:
            results_diff.append((q, work, err_diff, problem_type, i, QI, eps))
        if err_alg is not None:
            results_alg.append((q, work, err_alg, problem_type, i, QI, eps))

    if separate_errors:
        if single_sweeper or len(QI_list) == 2:
            work_plotter_diff = Plotter(nrows=1, ncols=1, figsize=(6, 6))
            work_plotter_alg = Plotter(nrows=1, ncols=1, figsize=(6, 6))
        else:
            num_QI = int(len(QI_list) / 2) if len(QI_list) % 2 == 0 else int((len(QI_list) + 1) / 2)

            work_plotter_diff = Plotter(nrows=num_QI, ncols=2, figsize=(12, num_QI * 6))
            work_plotter_alg = Plotter(nrows=num_QI, ncols=2, figsize=(12, num_QI * 6))

        for q_ind, work, err_diff, problem_type, i, QI, eps in results_diff:
            res = getMarker(problem_type, i, QI)

            color = getColorQI(QI) if single_sweeper else getColor(problem_type, i, QI)
            label = f"{QI}" if single_sweeper else getLabel(problem_type, eps, QI)
            linestyle = get_linestyle_QI(QI) if single_sweeper else get_linestyle(problem_type, QI)
            marker = getMarkerQI(QI) if single_sweeper else res["marker"]
            markersize = res["markersize"]

            work_plotter_diff = plot_result(work_plotter_diff, work, err_diff, q_ind, color, marker, markersize, linestyle, label)

        finalize_plot(
            case,
            num_nodes,
            problem_name,
            QI_list,
            work_plotter_diff,
            single_sweeper,
            solver_type,
            error_type,
            label_error="_diff",
            log_time=log_time,
            separate_errors=separate_errors,
        )

        for q_ind, work, err_alg, problem_type, i, QI, eps in results_alg:
            res = getMarker(problem_type, i, QI)

            color = getColorQI(QI) if single_sweeper else getColor(problem_type, i, QI)
            label = f"{QI}" if single_sweeper else getLabel(problem_type, eps, QI)
            linestyle = get_linestyle_QI(QI) if single_sweeper else get_linestyle(problem_type, QI)
            marker = getMarkerQI(QI) if single_sweeper else res["marker"]
            markersize = res["markersize"]

            work_plotter_alg = plot_result(work_plotter_alg, work, err_alg, q_ind, color, marker, markersize, linestyle, label)

        finalize_plot(
            case,
            num_nodes,
            problem_name,
            QI_list,
            work_plotter_alg,
            solver_type,
            single_sweeper,
            error_type,
            label_error="_alg",
            log_time=log_time,
            separate_errors=separate_errors,
        )

    else:
        if single_sweeper or len(QI_list) == 2:
            work_plotter = Plotter(nrows=1, ncols=1, figsize=(6, 6))
        else:
            num_QI = int(len(QI_list) / 2) if len(QI_list) % 2 == 0 else int((len(QI_list) + 1) / 2)

            work_plotter = Plotter(nrows=num_QI, ncols=2, figsize=(12, num_QI * 6))

        for q, work, err, problem_type, i, QI, eps in results:
            res = getMarker(problem_type, i, QI)

            color = getColorQI(QI) if single_sweeper else getColor(problem_type, i, QI)
            label = f"{QI}" if single_sweeper else getLabel(problem_type, eps, QI)
            linestyle = get_linestyle_QI(QI) if single_sweeper else get_linestyle(problem_type, QI)
            marker = getMarkerQI(QI) if single_sweeper else res["marker"]
            markersize = res["markersize"]

            plot_result(work_plotter, work, err, q, color, marker, markersize, linestyle, label)

        finalize_plot(
            case,
            num_nodes,
            problem_name,
            QI_list,
            work_plotter,
            single_sweeper,
            solver_type,
            error_type=error_type,
            log_time=log_time,
            separate_errors=separate_errors,
        )

def finalize_plot(
        k,
        num_nodes,
        problem_name,
        QI_list,
        work_plotter,
        single_sweeper,
        solver_type,
        error_type="global",
        label_error="",
        log_time=True,
        separate_errors=False,
    ):
    r"""
    Finalizes the plot with labels, titles, limits for y-axis and scaling, and legend. The plot is
    then stored with fixed filename.

    Parameters
    ----------
    dt : float
        Time step size.
    k : int
        Case number.
    num_nodes_list : list
        List contains different number of collocation nodes.
    problem_name : str
        Contains the name of the problem considered.
    QI_list : list
        Contains different :math:`Q_\Delta`.
    work_plotter : pySDC.projects.DAE.run.utils.Plotter
        Plotter class.
    """

    bbox_position = {1: -0.17, 2: -0.17, 3: -0.25, 4: -0.05, 5: -0.05, 6: -0.17, 11: -0.05, 14: -0.05, 15: -0.05}

    for q, QI in enumerate(QI_list):
        q_ind = 0 if single_sweeper or len(QI_list) == 2 else q

        if log_time:
            work_plotter.set_xlabel("wall-clock time", subplot_index=q_ind)
        else:
            work_plotter.set_xlabel("number of inner iterations", subplot_index=q_ind)

        if not single_sweeper:
            work_plotter.set_title(rf"$Q_\Delta=${QI}", subplot_index=q_ind, fontsize=20)

        # work_plotter.set_xlim((8e-3, 2e1), subplot_index=q)
        work_plotter.set_xscale(scale="log", subplot_index=q_ind)

        # work_plotter.set_ylim((1e-14, 1e0), subplot_index=q)
        work_plotter.set_yscale(scale="log", subplot_index=q_ind)

    if log_time:
        work_plotter.sync_xlim(min_x_set=1e-3)
    else:
        work_plotter.sync_xlim(min_x_set=1e0)

    work_plotter.sync_ylim(min_y_set=1e-16)

    if not separate_errors:
        work_plotter.set_ylabel(f"{error_type} error")
    else:
        if label_error == "_diff":
            work_plotter.set_ylabel(f"{error_type} error in y")
        elif label_error == "_alg":
            work_plotter.set_ylabel(f"{error_type} error in z")

    work_plotter.set_grid()

    if not single_sweeper:
        work_plotter.adjust_layout(num_subplots=len(QI_list))

    bbox_pos = bbox_position[k] - 0.2 if single_sweeper else bbox_position[k]
    ncol = 2 if single_sweeper else 4

    work_plotter.set_shared_legend(loc="lower center", bbox_to_anchor=(0.5, bbox_pos), ncol=ncol, fontsize=22)
    print("save")
    plot_type = "wallclocktime_error" if log_time else "work_error"
    filename = "data" + "/" + f"{problem_name}" + "/" + f"{plot_type}_case{k}_{num_nodes=}_{solver_type}{label_error}.png"
    try:
        work_plotter.save(filename)
    except (OverflowError, ValueError):
        for line in work_plotter.axes.get_lines():
            ydata = line.get_ydata()
            print("Contains inf:", np.any(np.isinf(ydata)))
            print("Contains nan:", np.any(np.isnan(ydata)))
            print("Contains zero:", np.any(ydata == 0))
        work_plotter.save(filename, uniform_size=False, use_constrained_layout=True)

def run(case, error_type, log_time, num_nodes, problem_name, QI_list, separate_errors, solver_type):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir = os.path.join("data", timestamp)
    os.makedirs(result_dir, exist_ok=True)

    latest_link = os.path.join("data", "latest")
    try:
        if os.path.lexists(latest_link):
            if os.path.islink(latest_link) or os.path.isfile(latest_link):
                os.remove(latest_link)
            elif os.path.isdir(latest_link):
                shutil.rmtree(latest_link)
        os.symlink(timestamp, latest_link, target_is_directory=True)
    except Exception as e:
        print(f"Warning: Could not set 'latest': {e}")

    problems = get_problem_cases(k=case, problem_name=problem_name)

    for problem_type,  eps_values in problems.items():
        for i, eps in enumerate(eps_values):
            for QI in QI_list:
                print(f"\nRunning {QI} with {problem_type} and eps={eps}...\n")

                args = base_args(log_time=log_time, problem_name=problem_name)
                args.update({
                    "QI": QI,
                    "problem_type": problem_type,
                    "eps": eps,
                    "problem_name": problem_name,
                    "case": str(case),
                    "num_nodes": str(num_nodes),
                    "solver_type": solver_type,
                    "error_type": error_type,
                })

                if separate_errors:
                    args["separate_errors"] = True

                args_list = []
                for k, v in args.items():
                    if isinstance(v, bool):
                        if v:
                            args_list.append(f"--{k}")
                    elif isinstance(v, list):
                        for item in v:
                            args_list.append(f"--{k}={item}")
                    else:
                        args_list.append(f"--{k}={v}")

                cmd = ["mpiexec", "-n", str(num_nodes), "python3", "run_single_qi.py"] + args_list
                env = os.environ.copy()
                env["PYSDC_RESULT_DIR"] = result_dir
                subprocess.run(cmd, check=True, env=env, close_fds=True)


# Script is started with "python3 work_error.py"
if __name__ == "__main__":
    QI_list = ["LU", "MIN-SR-NS"]#["IE", "LU", "MIN-SR-S", "MIN-SR-NS", "Picard"]
    # problem_name = "MICHAELIS-MENTEN"
    problem_name = "LINEAR-TEST"
    # problem_name = "ANDREWS-SQUEEZER"
    case = 15#4
    num_nodes = 3
    solver_type = "direct" #"hybr"
    error_type = "global"
    separate_errors = False  # True
    log_time = True # True

    run(case, error_type, log_time, num_nodes, problem_name, QI_list, separate_errors, solver_type)

    plot_work_error(case, error_type, log_time, num_nodes, problem_name, QI_list, separate_errors, solver_type)
