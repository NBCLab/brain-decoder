import argparse
import os.path as op

from braindec.dataset import _neurostore_to_nimare


def _get_parser():
    parser = argparse.ArgumentParser(description="Run gradient-decoding workflow")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        required=True,
        help="Path to project directory",
    )
    return parser


def main(project_dir):
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    neurostore_dir = op.join(data_dir, "pubmed")

    dset = _neurostore_to_nimare(neurostore_dir)
    dset.save(op.join(data_dir, "dset-pubmed_nimare.pkl"))


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
