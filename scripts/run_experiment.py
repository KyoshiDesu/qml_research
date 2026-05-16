from pathlib import Path

from qsvm_vm_compare.config import ExperimentConfig
from qsvm_vm_compare.experiment import run_experiment


def main() -> None:
    config_path = Path("configs/default_experiment.json")

    import argparse

    parser = argparse.ArgumentParser(description="Run the QSVM versus classical SVM benchmark.")
    parser.add_argument(
        "--config",
        type=Path,
        default=config_path,
        help="Path to the JSON experiment config.",
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config)
    result_path, results = run_experiment(config)

    print(f"Saved benchmark results to: {result_path}")
    print(f"Classical accuracy: {results['classical']['metrics']['accuracy']:.4f}")
    print(f"Quantum accuracy:   {results['quantum']['metrics']['accuracy']:.4f}")


if __name__ == "__main__":
    main()

