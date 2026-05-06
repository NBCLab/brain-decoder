"""Sphinx configuration for braindec documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "braindec"
copyright = "2025, Braindec developers"
author = "Braindec developers"

try:
    from braindec._version import __version__
    release = __version__
except ImportError:
    release = "unknown"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_argparse",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "nibabel": ("https://nipy.org/nibabel", None),
    "nilearn": ("https://nilearn.github.io/stable", None),
    "nimare": ("https://nimare.readthedocs.io/en/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# sphinx-gallery configuration
# The first_notebook_cell is injected verbatim as the first code cell of
# every generated .ipynb, so Colab users can install the package before
# running any example code.
_COLAB_INSTALL_CELL = """\
# Install braindec (this cell is only needed on Google Colab).
import importlib, subprocess, sys

if importlib.util.find_spec("braindec") is None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "braindec[plotting]"],
        check=True,
    )
"""

sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    # Only process files whose names start with two digits.
    "filename_pattern": r"/\d{2}_",
    # Inject the Colab install cell at the top of every generated notebook.
    "first_notebook_cell": _COLAB_INSTALL_CELL,
    # Set to True locally to execute examples and capture outputs.
    # On ReadTheDocs the examples are not executed (too slow / GPU-dependent).
    "plot_gallery": os.environ.get("BRAINDEC_BUILD_GALLERY", "0") == "1",
    "remove_config_comments": True,
    "show_memory": False,
    "doc_module": ("braindec",),
    "reference_url": {"braindec": None},
    "backreferences_dir": "gen_modules/backreferences",
    "image_scrapers": ("matplotlib",),
    "default_thumb_file": "../NiCLIP.png",
}
