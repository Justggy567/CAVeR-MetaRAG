import argparse

from src.RQ1_code import run_benchmark as run_rq1
# from RQ1_code import run_benchmark as run_rq1
from src.RQ2_gemma3_deepseek import run_benchmark as run_rq2
from src.RQ3_3types import run_experiment_single_model as run_rq3
from src.RQ3_structure_aware_data_statistics import run_combined_experiment as run_rq3_compare


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=[
            # "rq1",
            # "rq2",
            # "rq3",
            # "rq3_compare"
        ]
    )

    args = parser.parse_args()

    print("=" * 60)
    print(f"Running Experiment: {args.experiment}")
    print("=" * 60)

    if args.experiment == "rq1":
        run_rq1()

    elif args.experiment == "rq2":
        run_rq2()

    elif args.experiment == "rq3":
        run_rq3()

    elif args.experiment == "rq3_compare":
        run_rq3_compare()


if __name__ == "__main__":
    main()