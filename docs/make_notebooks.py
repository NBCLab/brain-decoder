"""Convert sphinx-gallery .py examples to committed .ipynb files for Colab.

Usage:
    python docs/make_notebooks.py
    python docs/make_notebooks.py --examples examples/02_niclip_demo.py

The generated notebooks are written to docs/auto_examples/ so that the Colab
badge URL in each example resolves to a real file on GitHub.  Commit the
output alongside the .py source.

The sphinx-gallery ``first_notebook_cell`` content (braindec install) is
injected as the first code cell of every notebook.
"""

import argparse
import json
from pathlib import Path

INSTALL_CELL_SOURCE = """\
# Install braindec (this cell is only needed on Google Colab).
import importlib, subprocess, sys

if importlib.util.find_spec("braindec") is None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "braindec[plotting]"],
        check=True,
    )
"""

INSTALL_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in INSTALL_CELL_SOURCE.rstrip().splitlines()],
}


def py_to_notebook(py_path: Path, out_path: Path) -> None:
    """Convert a sphinx-gallery .py file to an .ipynb, prepending the install cell."""
    from nbformat.v4 import new_code_cell
    import jupytext

    notebook = jupytext.read(py_path, fmt="py:percent")
    notebook.cells.insert(0, new_code_cell(INSTALL_CELL_SOURCE))
    jupytext.write(notebook, out_path, fmt="ipynb")
    print(f"  {py_path.name} → {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples",
        nargs="*",
        default=None,
        help="Specific .py files to convert.  Defaults to all examples/NN_*.py files.",
    )
    parser.add_argument(
        "--stage",
        action="store_true",
        help="Run 'git add' on generated notebooks (used by the pre-commit hook).",
    )
    opts = parser.parse_args(argv)

    repo_root = Path(__file__).parent.parent
    out_dir = repo_root / "docs" / "auto_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    if opts.examples:
        sources = [Path(p) for p in opts.examples]
    else:
        sources = sorted((repo_root / "examples").glob("[0-9][0-9]_*.py"))

    if not sources:
        print("No example files found.")
        return

    try:
        import jupytext  # noqa: F401
    except ImportError:
        raise SystemExit("jupytext is required: pip install jupytext")

    generated = []
    for src in sources:
        dest = out_dir / src.with_suffix(".ipynb").name
        py_to_notebook(src, dest)
        generated.append(dest)

    if opts.stage and generated:
        import subprocess
        subprocess.run(["git", "add", *generated], check=True)
        print("Staged generated notebooks.")
    else:
        print(f"\nDone. Commit the files in {out_dir} to enable the Colab badge links.")


if __name__ == "__main__":
    main()
