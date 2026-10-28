"""Check the completeness and hashes of a recorded-data release."""

from argparse import _SubParsersAction as SubParsersAct

from ho_optim_milp.dataset_pipeline import validate_dataset


def add_parser(subparsers: SubParsersAct) -> None:
    parser = subparsers.add_parser("validate_dataset")
    parser.add_argument("--dataset-root", required=True)
    parser.set_defaults(func=main)


def main(dataset_root: str, **_: object) -> int:
    result = validate_dataset(dataset_root)
    print(f"Validated {result['artifact_count']} dataset artifacts.")
    return 0
