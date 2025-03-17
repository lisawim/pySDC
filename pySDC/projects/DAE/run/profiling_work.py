from mpi4py import MPI
import pandas as pd
import cProfile
import pstats
import pathlib
import os

from pySDC.projects.DAE.run.wallclocktime_error import compute_work_vs_error
from pySDC.projects.DAE import Plotter


EXCLUDED_FUNCTIONS = [
    "__init__",
    "__generate_hierarchy",
    "_call_with_frames_removed",
    "exec_module",
    "_load_unlocked",
    "_find_and_load_unlocked",
    "_find_and_load",
    "<module>",
    "compute_work_vs_error",
    "run_serial_test",
    "run_parallel_test",
    "computeSolution",
    "run",
    "pfasst",
    "check_support_sve",
    "<method 'read' of '_io.BufferedReader' objects",
    "<built-in method _imp.create_dynamic>",
    "create_module",
    "get_data",
    "module_from_spec",
    "_handle_fromlist",
    "get_code",
    "function Unpickler.find_class at 0x75ca8a4a2a70>",
    "find_class",
    "function Unpickler.load at 0x75ca8a4a2b90>",
    "load",
    "<built-in method builtins.__import__>",
    "<built-in method builtins.exec>",
    "<method 'disable' of '_lsprof.Profiler' objects>",  # Internal profiler function
]

EXCLUDED_PATTERNS = [
    r".*profiling.*",  # Matches any function containing "profiling"
    r".*helper_function.*",  # Matches functions like "helper_function_1", "helper_function_XYZ"
    r"<function Unpickler.*>",
    r"<.*built-in.*>",
    r"_call*",
    r"_find*",
    r"_load*",
    r".*load.*",
    r".*__init__.*",
    r".*exec_module.*",
]

INCLUDED_FUNCTIONS = [
    "solve_system",
    "u_exact",
    "integrate",
    "update_nodes",
]


def finalize_plot(dt, k, num_nodes, profiling_plotter, problem_name, QI, size, yticks, problem_type_label):
    fontsize = 34 if QI == "MIN-SR-S" else 18
    if QI == "MIN-SR-S":
        for r in range(size):
            profiling_plotter.set_title(f"{QI}-{problem_type_label} - Rank {r}", subplot_index=r, fontsize=24)

            profiling_plotter.set_xlabel("Cumulative Execution Time (seconds)", subplot_index=r)
            profiling_plotter.set_ylabel("Function", subplot_index=r)

    else:
        profiling_plotter.set_title(f"{QI}-{problem_type_label}", subplot_index=0, fontsize=24)

        profiling_plotter.set_xlabel("Cumulative Execution Time (seconds)", subplot_index=0)
        profiling_plotter.set_ylabel("Function", subplot_index=0)

    profiling_plotter.set_yticks(range(len(yticks)), labels=yticks, fontsize=fontsize)

    if QI == "MIN-SR-S":
        profiling_plotter.adjust_layout(num_subplots=size)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"profiling_work_case{k}_{num_nodes=}_{QI}_{dt=}.png"
    profiling_plotter.save(filename)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    problem_name = "MICHAELIS-MENTEN"

    path_name = f"data/{problem_name}/profiling/"

    if rank == 0:
        pathlib.Path(path_name).mkdir(parents=True, exist_ok=True)

    QI_list = ["IE", "LU", "MIN-SR-S"]
    num_nodes = size

    problem_type_labels = {10: r"$\mathtt{SDC-E}$", 11: r"$\mathtt{SDC-C}$", 12: r"$\mathtt{FI-SDC}$", 13: r"$\mathtt{SI-SDC}$"}

    case = 11

    dt_list = [1e-6]

    for QI in QI_list:
        is_parallel = QI == "MIN-SR-S"  # Parallel flag

        # Create a profiler
        profiler = cProfile.Profile()
        profiler.enable()

        # compute_work_vs_error(case, comm, num_nodes, rank, problem_name, [QI], do_plotting=False, dt_list=dt_list)
        print(f"Hello from {rank=}")

        profiler.disable()

        profile_filename = f"{path_name}profile_{QI}_rank_{rank}_{num_nodes}.prof"
        # Save profiling data (binary format)
        if is_parallel or rank == 0:
            profile_filename = f"{path_name}profile_{QI}_rank_{rank}_{num_nodes}.prof"
            try:
                profiler.dump_stats(profile_filename)  # ✅ Save in binary format
            except Exception as e:
                print(f"❌ Error saving profiling data on Rank {rank}, QI {QI}: {e}")

    comm.Barrier()  # Synchronize all ranks before visualization

    # Rank 0 collects and plots results
    if rank == 0:
        # Read profiling data from all relevant ranks
        data = []

        for QI in QI_list:
            is_parallel = QI == "MIN-SR-S"

            ranks_to_check = range(size) if is_parallel else [0]

            for r in ranks_to_check:
                profile_filename = f"{path_name}profile_{QI}_rank_{r}_{num_nodes}.prof"

                if os.path.exists(profile_filename):
                    file_size = os.path.getsize(profile_filename)
                    print(f"✔ Found: {profile_filename} ({file_size} bytes)")
                else:
                    print(f"❌ Missing: {profile_filename}")

        # Load and analyze profiling data
        for QI in QI_list:
            is_parallel = QI == "MIN-SR-S"
            ranks_to_check = range(size) if is_parallel else [0]

            for r in ranks_to_check:
                profile_filename = f"{path_name}profile_{QI}_rank_{r}_{num_nodes}.prof"

                # Ensure file exists and is not empty
                if os.path.exists(profile_filename) and os.path.getsize(profile_filename) > 0:
                    try:
                        stats = pstats.Stats(profile_filename)
                        stats.strip_dirs().sort_stats("cumulative")

                        # Extract function names and cumulative execution times
                        for func, stat in stats.stats.items():
                            func_name = f"{func[2]} ({func[0]}:{func[1]})"
                            cumulative_time = stat[3]  # Cumulative time spent in function
                            data.append((QI, r, func_name, cumulative_time))

                    except Exception as e:
                        print(f"❌ Error reading profiling file {profile_filename}: {e}")
                else:
                    print(f"❌ Warning: Profile file {profile_filename} is missing or empty.")

        # Convert to DataFrame
        df = pd.DataFrame(data, columns=["QI", "Rank", "Function", "Cumulative Time"])
        df = df.sort_values(by=["QI", "Rank", "Cumulative Time"], ascending=[True, True, False])

        print("\n==== Available Functions Before Filtering ====")
        print(df["Function"].unique())  # Print all unique functions before filtering
        print("\n============================================\n")

        # ✅ If INCLUDED_FUNCTIONS is set, use only these functions
        if INCLUDED_FUNCTIONS:
            df = df[df["Function"].isin(INCLUDED_FUNCTIONS)]
            print(f"\n✅ Using ONLY selected functions: {INCLUDED_FUNCTIONS}\n")
        # else:
        #     # ✅ Otherwise, apply exclusion filters
        #     df = df[~df["Function"].isin(EXCLUDED_FUNCTIONS)]
            
        #     pattern = "|".join(EXCLUDED_PATTERNS)
        #     df = df[~df["Function"].str.match(pattern, na=False, case=False)]

        #     print("\n✅ Exclusion applied: Removing unwanted functions & patterns\n")

        for QI in QI_list:
            is_parallel = QI == "MIN-SR-S"

            nrows = 2 if is_parallel else 1
            ncols = nrows
            figsize = (48, 24) if is_parallel else (20, 12)

            profiling_plotter = Plotter(nrows=nrows, ncols=ncols, figsize=figsize)

            df_qi = df[df["QI"] == QI]  # Show top 5 functions per QI

            if df_qi.empty:
                print(f"❌ Warning: No profiling data found for QI={QI}, skipping plot.")
                continue

            # Ensure every rank is included
            ranks_to_plot = range(size) if is_parallel else [0]

            for r in ranks_to_plot:
                df_rank = df_qi[df_qi["Rank"] == r].head(17)

                subplot_index = r % (nrows * ncols)  # Ensure valid subplot index
                profiling_plotter.barh(
                    df_rank["Function"],
                    df_rank["Cumulative Time"],
                    subplot_index=subplot_index,
                )

            finalize_plot(dt_list[-1], case, num_nodes, profiling_plotter, problem_name, QI, size, df_rank["Function"], problem_type_labels[case])

    MPI.Finalize()
